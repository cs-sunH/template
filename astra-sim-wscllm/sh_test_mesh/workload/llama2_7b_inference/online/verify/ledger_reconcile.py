#!/usr/bin/env python3
"""ledger_reconcile.py -- 结束总账核对(方案 §6.2 操作 3 / contract ⑥;
阶段 7 §10.1 扩展为逐层对账)。

Python KV/排队账本 vs C++ 执行事实,逐 request/stage 对平。输入(同一
strategy 感知运行目录):

  --bridge-dir <run>/bridge   request_*.json(C++ 完成事实 + completed_nodes
                              逐节点完成明细;phase-7 §10.3 起 response 文件
                              消费即删,committed 侧数据源迁移至此)
                              graph_batch_digests.jsonl(批级 committed
                              摘要:watch_count / ranks / node_count)
                              online_decision_log.jsonl(每 request 恰
                              prefill/decode/completion 三条决策;completion
                              tick = KV 生命周期终止时刻)
                              ledger.jsonl(Python 分层账本导出)
                              sensing_query_log.jsonl(决策边界两层剩余负载
                              查询快照)
                              cpp.log 的 phase-5 commit counters 与
                              phase-4 end audit(committed/watch/kv 权威计数)
  --manifest <ET_DIR>/manifest.json   全部请求事实(1177 requests)
  --cpp-log <run>/cpp.log     C++ 门计数器(可选;存在则核对 completed)
  --expected-requests N       输入期望请求数(R0a;缺省 1177 = 20.csv
                              前 30s 验收值,跨输入复用时按物化实测改传)
  --expected-accepted-sessions N
                              输入期望 accepted 会话数(R0e;缺省 112
                              = 30s 验收值,同上)
  --report <path>.md          对账报告输出(markdown)

对平项(全部硬断言;任一失配即"不平"并逐项列差异与证据):

  R0 集合对平:manifest 1177 == Python completed-unreconciled 1177
     == C++ REQUEST_COMPLETE 事实 1177 == cpp.log completed 1177;
     prefill/decode 完成事实各 1177;同一 request/stage 零重复完成。
  R1 逐 request 生命周期对平:admitted_tick <= first_commit_tick
     <= prefill_complete_tick <= decode_complete_tick == completed_tick
     (decode 与 request-complete 同 fire 同 epoch,必等);
     generation:prefill=0 / decode=1 / request-complete=1。
  R2 committed 节点对平(watch 成员账本,phase-7 契约):每 (request, stage)
     恰注册一次 watch —— C++ total_watches == completed_nodes 对计数 ==
     ledger stage 对计数 == sum(digests.watch_count);批级摘要合法(watch
     每批 watch_count ∈ [0, len(ranks)] 且与 ranks 空性一致、ranks ⊆ [0,53]、sum(node_count) == C++ total_nodes);
     watch 生命周期干净收尾(phase-4 end audit 全零)。
  R3 KV 账本对平(phase-7 契约):每 request 恰一条 completion 决策
     (online_decision_log, tick = KV 生命周期终止时刻)且 C++
     total_kv_actions >= 请求数(每 request ≥ 1 动作的计数级保证);
     completion tick == ledger completed_tick;prefill<=decode<=completion
     顺序;kv_instance_after_completion ∈ [0,53];运行结束 active=0。
  R4 排队账本对平:运行结束 admitted/committed 层已清空(ledger.jsonl 无
     admitted/committed 条目);最后一决策边界 admitted_count == 0。
  R5 injected-unfinished 对平(sensing_query_log):末决策边界残差 = 终请求
     decode end-barrier 控制节点(1 字节 all_reduce,无 compute、无时长;
     最终 watch 命中时仍在途,运行结束前完成——C++ 干净退出 active=0
     即证),断言残差非空且全部为 barrier 签名;注入峰值(节点数,跨
     rank/代求和)≤ committed 节点数(以 completed_nodes 逐对计数为
     上限;提交后逐批流入,完成时归零)。
  R6 ready 层对平(§10.1):每边界 ready(就绪可服务排队成员,策略 override
     视图)⊆ admitted(键集)且 ready ∩ issued == ∅;末决策边界 ready==0。
  R7 issued 层对平(§10.1,核心):每边界 Python issued 键集(已发射未完成;
     emit 登记、completion 核销于策略完成处理前)⊆ C++ injected-unfinished
     per_request 键集(同 delivery 同 tick 摘要);only-C++ 残差全部为
     end-barrier 控制尾节点签名(watch 命中即报阶段完成、1 字节 all_reduce
     尾节点仍在途的已知窗口,与 R5 同源);末决策边界 issued==0。
  R8 network pending/active 计数对账(§10.1):C++ 摘要内部一致(每 rank
     node_count >= in_flight_node_count + free_node_count,剩余 = 依赖未
     满足的未发射中间节点);在飞节点数(跨 rank)>= issued request 数(每
     在飞 request 至少 1 节点);峰值统计列示。

八层账本口径(§10.1):admitted / committed / ready / issued / network
pending/active 参与对平;remote FIFO 与 local HBM 为"不适用"显式占位
(wscllm 无远端内存后端 FIFO;local HBM 计时模型已激活——comm 边端点
计费 hbm_charge 在案(裁决 #35),缺的是独立 local-HBM 节点账本,故
该层无 job 账本可对平),报告中列示不参与对平。

退出码:0 = 对平;1 = 存在失配(报告逐项列出差异与证据)。"""

import argparse
import gzip
import json
import os
import re
import sys

_ONLINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ONLINE_DIR not in sys.path:
    sys.path.insert(0, _ONLINE_DIR)
from bridge_request_journal import iter_request_records  # noqa: E402


# ---------------------------------------------------------------------------
# 输入加载
# ---------------------------------------------------------------------------

def _artifact_path(bridge_dir, name):
    """Resolve live bridge artifacts first, then runner-archived results/."""
    live = os.path.join(bridge_dir, name)
    if os.path.isfile(live):
        return live
    run_dir = os.path.dirname(os.path.abspath(bridge_dir))
    archived = os.path.join(run_dir, "results", name)
    if os.path.isfile(archived):
        return archived
    return live


def _read_cpp_log(cpp_log_path):
    """Read cpp.log or its transparent post-archive cpp.log.gz form."""
    if not cpp_log_path:
        return None
    path = cpp_log_path
    if not os.path.isfile(path) and not path.endswith(".gz") \
            and os.path.isfile(path + ".gz"):
        path += ".gz"
    if not os.path.isfile(path):
        return None
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as source:
        return source.read()


def _iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as source:
        manifest = json.load(source)
    request_ids = sorted(
        record["request_id"] for record in manifest["requests"])
    return request_ids


def load_cpp_facts(bridge_dir):
    """从 request_*.json 收集 C++ 执行事实:
      completions: request_id -> {"prefill": {tick, generation, member_count},
                                  "decode": {...}, "request_complete": {...}}
      arrival_epochs / completion_epochs 计数(信息)。
    同一 (request, stage, generation) 重复出现 = 失配。"""
    completions = {}
    arrival_epochs = 0
    for seq, req in iter_request_records(bridge_dir):
        arrival_epochs += len(req.get("arrivals") or [])
        for group in req.get("completed_groups") or []:
            request_id = group["request_id"]
            stage = group["stage"]
            fact = {
                "tick": req["tick"],
                "generation": group.get("generation", 0),
                "member_count": group.get("node_count", 0),
                "delivery_sequence": seq,
            }
            entry = completions.setdefault(request_id, {})
            key = stage if stage else "request_complete"
            if key in entry:
                raise AssertionError(
                    "C++ fact duplicated: request {!r} stage {!r} appears "
                    "twice in completed_groups".format(request_id, stage))
            entry[key] = fact
    return completions, arrival_epochs


def load_committed(bridge_dir):
    """phase-7 契约(request 文件 = C++ 完成事实 + 逐节点完成明细;
    response 文件消费即删,committed 侧数据源迁移至 request 文件):
      committed_counts: (request_id, stage) -> completed_nodes 总数
      committed_ranks:  (request_id, stage) -> rank 集合(completed_nodes)
      group_pairs:      completed_groups 的 (request_id, stage) 对集合
                        (stage 非空为 prefill/decode 组;stage=='' 为
                        request_complete 控制组,无 completed_nodes)
      completed_nodes_total: 全部 completed_nodes 条数
    completed_nodes 即旧 response 文件 "nodes"(逐节点 rank/node_id),
    terminal_status=0 表示正常完成。"""
    committed_counts = {}
    committed_ranks = {}
    group_pairs = set()
    completed_nodes_total = 0
    for seq, req in iter_request_records(bridge_dir):
        for group in req.get("completed_groups") or []:
            group_pairs.add((group["request_id"], group["stage"]))
        for node in req.get("completed_nodes") or []:
            key = (node.get("request_id", ""), node.get("stage", ""))
            committed_counts[key] = committed_counts.get(key, 0) + 1
            committed_ranks.setdefault(key, set()).add(int(node["rank"]))
            completed_nodes_total += 1
    return committed_counts, committed_ranks, group_pairs, \
        completed_nodes_total


def load_digests(bridge_dir):
    """graph_batch_digests.jsonl(批级 committed 摘要):每批 watch_count
    ∈ [0, len(ranks)] 且 watch_count==0 ⇔ ranks 空,ranks = 批内 committed
    rank 集,node_count = 批节点总数。
    sum(node_count) == C++ total_nodes;sum(watch_count) == C++ total_watches
    (每 (request, stage) 恰注册一次 watch 的批级等价)。"""
    digests = []
    watch_total = 0
    node_total = 0
    for row in _iter_jsonl(_artifact_path(
            bridge_dir, "graph_batch_digests.jsonl")):
        digests.append(row)
        watch_total += row.get("watch_count", 0)
        node_total += row.get("node_count", 0)
    return digests, watch_total, node_total


def load_decision_log(bridge_dir):
    """online_decision_log.jsonl:每 request 恰 prefill/decode/completion
    三条决策(kind);completion 行 tick 即 KV 生命周期终止时刻。"""
    by_kind = {"prefill": {}, "decode": {}, "completion": {}}
    kind_counts = {}
    rows = 0
    for row in _iter_jsonl(_artifact_path(
            bridge_dir, "online_decision_log.jsonl")):
        rows += 1
        kind = row.get("kind")
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        if kind in by_kind:
            by_kind[kind][row["request_id"]] = row
    return by_kind, kind_counts, rows


def load_cpp_audit(cpp_log_path):
    """cpp.log 的 phase-5 commit counters 与 phase-4 end audit(phase-7
    新契约的 committed/watch/kv 权威计数)。"""
    text = _read_cpp_log(cpp_log_path)
    if text is None:
        return None
    out = {}
    m5 = re.search(
        r"\[online\] phase-5 commit counters: graph_batch_count=(\d+) "
        r"single_node_bridge_count=(\d+) total_nodes=(\d+) "
        r"avg_nodes_per_batch=[0-9.]+ max_nodes_per_batch=(\d+) "
        r"total_watches=(\d+) total_assignments=(\d+) "
        r"total_kv_actions=(\d+) total_future_alarms=(\d+)", text)
    if m5:
        out.update({
            "graph_batch_count": int(m5.group(1)),
            "single_node_bridge_count": int(m5.group(2)),
            "total_nodes": int(m5.group(3)),
            "max_nodes_per_batch": int(m5.group(4)),
            "total_watches": int(m5.group(5)),
            "total_assignments": int(m5.group(6)),
            "total_kv_actions": int(m5.group(7)),
            "total_future_alarms": int(m5.group(8)),
        })
    m4 = re.search(
        r"\[online\] phase-4 end audit: watch_registry_size=(\d+) "
        r"watch_stale=(\d+) completed_facts_residual=(\d+) "
        r"affected_ranks_residual=(\d+)", text)
    if m4:
        out.update({
            "watch_registry_size": int(m4.group(1)),
            "watch_stale": int(m4.group(2)),
            "completed_facts_residual": int(m4.group(3)),
            "affected_ranks_residual": int(m4.group(4)),
        })
    return out or None


def load_python_ledger(bridge_dir):
    """ledger.jsonl:request_id -> {admitted, committed,
    completed_unreconciled}(各层可能缺省)。"""
    ledger = {}
    for row in _iter_jsonl(_artifact_path(bridge_dir, "ledger.jsonl")):
        ledger[row["request_id"]] = row.get("layers", {})
    return ledger


def load_sensing_query_log(bridge_dir):
    return list(_iter_jsonl(
        _artifact_path(bridge_dir, "sensing_query_log.jsonl")))


def load_cpp_counters(cpp_log_path):
    """cpp.log 的 gate/service counters:[online] service counters:
    accepted=112 completed=1177 active=0 pending_alarm=0"""
    text = _read_cpp_log(cpp_log_path)
    if text is None:
        return None
    match = re.search(
        r"\[online\] service counters: accepted=(\d+) completed=(\d+)"
        r" active=(\d+) pending_alarm=(\d+)", text)
    if not match:
        return None
    return {
        "accepted": int(match.group(1)),
        "completed": int(match.group(2)),
        "active": int(match.group(3)),
        "pending_alarm": int(match.group(4)),
    }


def _seq_files(bridge_dir, prefix):
    seqs = []
    for name in os.listdir(bridge_dir):
        if name.startswith(prefix) and name.endswith(".json"):
            body = name[len(prefix):-len(".json")]
            if body.isdigit():
                seqs.append(int(body))
    return sorted(seqs)


# ---------------------------------------------------------------------------
# 对账
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="phase-3 ledger reconciliation (strategy sensing run)")
    parser.add_argument("--bridge-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cpp-log", default=None)
    parser.add_argument(
        "--expected-requests", type=int, default=1177,
        help="输入期望请求数(R0a;缺省 1177 = 20.csv 前 30s 验收值,"
             "跨输入复用时按物化实测改传)")
    parser.add_argument(
        "--expected-accepted-sessions", type=int, default=112,
        help="输入期望 accepted 会话数(R0e;缺省 112 = 30s 验收值,同上)")
    parser.add_argument("--report", default=None,
                        help="markdown report path (default: stdout)")
    args = parser.parse_args(argv)

    failures = []  # (item, evidence)

    def check(item, cond, evidence):
        if not cond:
            failures.append((item, evidence))
            return False
        return True

    # ---- R0 集合对平 ----
    manifest_ids = load_manifest(args.manifest)
    cpp_completions, arrival_epochs = load_cpp_facts(args.bridge_dir)
    committed_counts, committed_ranks, group_pairs, completed_nodes_total = \
        load_committed(args.bridge_dir)
    digests, watch_total, digest_node_total = load_digests(args.bridge_dir)
    decision_log, decision_kind_counts, decision_rows = \
        load_decision_log(args.bridge_dir)
    python_ledger = load_python_ledger(args.bridge_dir)
    query_log = load_sensing_query_log(args.bridge_dir)
    cpp_counters = load_cpp_counters(args.cpp_log)
    cpp_audit = load_cpp_audit(args.cpp_log)

    manifest_count = len(manifest_ids)
    check("R0a", manifest_count == args.expected_requests,
          "manifest requests = {} (expected {})".format(
              manifest_count, args.expected_requests))

    unreconciled = {
        request_id: layers.get("completed_unreconciled")
        for request_id, layers in python_ledger.items()
        if layers.get("completed_unreconciled") is not None
    }
    check("R0b", len(unreconciled) == manifest_count,
          "Python completed-unreconciled = {} (expected {})".format(
              len(unreconciled), manifest_count))
    extra = set(unreconciled) - set(manifest_ids)
    missing = set(manifest_ids) - set(unreconciled)
    check("R0b1", not extra,
          "Python ledger requests outside manifest: {!r}".format(
              sorted(extra)))
    check("R0b2", not missing,
          "manifest requests missing from Python ledger: {!r}".format(
              sorted(missing)))

    prefill_facts = {
        rid: entry.get("prefill") for rid, entry in cpp_completions.items()}
    decode_facts = {
        rid: entry.get("decode") for rid, entry in cpp_completions.items()}
    req_facts = {
        rid: entry.get("request_complete")
        for rid, entry in cpp_completions.items()}
    check("R0c", len(req_facts) == manifest_count,
          "C++ REQUEST_COMPLETE facts = {} (expected {})".format(
              len(req_facts), manifest_count))
    check("R0c1", len(prefill_facts) == manifest_count,
          "C++ prefill completion facts = {} (expected {})".format(
              len(prefill_facts), manifest_count))
    check("R0c2", len(decode_facts) == manifest_count,
          "C++ decode completion facts = {} (expected {})".format(
              len(decode_facts), manifest_count))
    check("R0c3", not (set(prefill_facts) - set(manifest_ids)),
          "prefill facts outside manifest: {!r}".format(
              sorted(set(prefill_facts) - set(manifest_ids))))
    check("R0c4", not (set(decode_facts) - set(manifest_ids)),
          "decode facts outside manifest: {!r}".format(
              sorted(set(decode_facts) - set(manifest_ids))))
    check("R0c5", not (set(req_facts) - set(manifest_ids)),
          "request-complete facts outside manifest: {!r}".format(
              sorted(set(req_facts) - set(manifest_ids))))

    if cpp_counters is not None:
        check("R0d", cpp_counters["completed"] == manifest_count,
              "cpp.log completed = {} (expected {})".format(
                  cpp_counters["completed"], manifest_count))
        check("R0e", cpp_counters["accepted"]
              == args.expected_accepted_sessions,
              "cpp.log accepted sessions = {} (expected {})".format(
                  cpp_counters["accepted"],
                  args.expected_accepted_sessions))

    # ---- R1 逐 request 生命周期对平 ----
    lifecycle_bad = []
    for request_id in manifest_ids:
        layers = python_ledger.get(request_id, {})
        unrecon = layers.get("completed_unreconciled") or {}
        admitted_tick = unrecon.get("admitted_tick")
        first_commit_tick = unrecon.get("first_commit_tick")
        completed_tick = unrecon.get("completed_tick")
        prefill = prefill_facts.get(request_id) or {}
        decode = decode_facts.get(request_id) or {}
        reqc = req_facts.get(request_id) or {}
        sequence_ok = (
            admitted_tick is not None
            and first_commit_tick is not None
            and completed_tick is not None
            and admitted_tick <= first_commit_tick
            and first_commit_tick <= prefill.get("tick", -1)
            and prefill.get("tick", -1) <= decode.get("tick", -1)
            and decode.get("tick", -1) == completed_tick)
        if not sequence_ok:
            lifecycle_bad.append({
                "request_id": request_id,
                "admitted_tick": admitted_tick,
                "first_commit_tick": first_commit_tick,
                "prefill_tick": prefill.get("tick"),
                "decode_tick": decode.get("tick"),
                "completed_tick": completed_tick,
            })
        gen_ok = (prefill.get("generation") == 0
                  and decode.get("generation") == 1
                  and reqc.get("generation") == 1)
        if not gen_ok:
            lifecycle_bad.append({
                "request_id": request_id,
                "generations": {
                    "prefill": prefill.get("generation"),
                    "decode": decode.get("generation"),
                    "request_complete": reqc.get("generation"),
                },
            })
    check("R1", not lifecycle_bad,
          "{} lifecycle violations (admitted<=commit<=prefill<=decode=="
          "complete, gen 0/1/1)".format(len(lifecycle_bad)))

    # ---- R2 committed 节点对平(watch 成员账本,phase-7 契约) ----
    # response_*.json 已随阶段 7 §10.3 改为消费即删;committed 侧数据源
    # 迁移到:(a) request 文件 completed_nodes(逐节点明细,即旧 response
    # "nodes";每 (request, stage) 对完成时恰覆盖其全部 committed 节点)、
    # (b) graph_batch_digests.jsonl(批级 committed 摘要:watch_count /
    # ranks / node_count)、(c) cpp.log phase-5 commit counters 与
    # phase-4 end audit(C++ 权威计数)。
    #   R2a 每 (request, stage) 恰注册一次 watch:C++ total_watches ==
    #       completed_nodes 对计数 == ledger stage 对计数 ==
    #       sum(digests.watch_count);
    #   R2b 批级摘要合法:watch 每批 ∈ [0, len(ranks)] 且与 ranks 空性一致、
    #   ranks ⊆ [0,53]、
    #       sum(node_count) == C++ total_nodes;
    #   R2c watch 生命周期干净收尾:phase-4 end audit 全零。
    watch_total_bad = []
    ledger_pairs = set()
    for rid, layers in python_ledger.items():
        for stage in ((layers.get("completed_unreconciled") or {})
                      .get("stages") or []):
            ledger_pairs.add((rid, stage["stage"]))
    committed_pairs = {key for key in committed_counts
                       if key[1] in ("prefill", "decode")}
    if cpp_audit is not None:
        if cpp_audit["total_watches"] != len(committed_pairs):
            watch_total_bad.append({
                "cpp_total_watches": cpp_audit["total_watches"],
                "completed_nodes_pairs": len(committed_pairs)})
        if cpp_audit["total_watches"] != len(ledger_pairs):
            watch_total_bad.append({
                "cpp_total_watches": cpp_audit["total_watches"],
                "ledger_stage_pairs": len(ledger_pairs)})
        if cpp_audit["total_watches"] != watch_total:
            watch_total_bad.append({
                "cpp_total_watches": cpp_audit["total_watches"],
                "digest_watch_sum": watch_total})
    elif len(committed_pairs) != len(ledger_pairs) \
            or len(committed_pairs) != watch_total:
        watch_total_bad.append({
            "completed_nodes_pairs": len(committed_pairs),
            "ledger_stage_pairs": len(ledger_pairs),
            "digest_watch_sum": watch_total})
    batch_watch_bad = []
    batch_rank_bad = []
    for digest in digests:
        wc = digest.get("watch_count", 0)
        ranks = digest.get("ranks") or []
        if not (isinstance(wc, int) and 0 <= wc <= len(ranks)
                and (wc == 0) == (len(ranks) == 0)):
            batch_watch_bad.append({
                "delivery_sequence": digest["delivery_sequence"],
                "watch_count": wc, "ranks_len": len(ranks)})
        if not all(0 <= rank <= 53 for rank in digest.get("ranks", [])):
            batch_rank_bad.append({
                "delivery_sequence": digest["delivery_sequence"],
                "ranks": digest.get("ranks")})
    node_total_bad = []
    if cpp_audit is not None and cpp_audit["total_nodes"] != \
            digest_node_total:
        node_total_bad.append({
            "cpp_total_nodes": cpp_audit["total_nodes"],
            "digest_node_sum": digest_node_total})
    audit_bad = []
    if cpp_audit is not None:
        for field in ("watch_registry_size", "watch_stale",
                      "completed_facts_residual",
                      "affected_ranks_residual"):
            if cpp_audit.get(field, 0) != 0:
                audit_bad.append({field: cpp_audit[field]})
    check("R2a", not watch_total_bad,
          "watch 注册总数失配(每 (request, stage) 恰一次): {}".format(
              watch_total_bad))
    check("R2b", not batch_watch_bad and not batch_rank_bad
          and not node_total_bad,
          "批级 committed 摘要失配: watch 每批 ∈[0,len(ranks)] 且空性一致 {} 条 / ranks 越界 "
          "{} 条 / node_count 总量 {} 条".format(
              len(batch_watch_bad), len(batch_rank_bad),
              len(node_total_bad)))
    check("R2c", not audit_bad,
          "watch 生命周期残留(phase-4 end audit 非零): {}".format(
              audit_bad))

    # ---- R3 KV 账本对平(phase-7 契约) ----
    # kv_actions 随 response 文件消费即删,逐事件明细不再落盘;KV 生命周期
    # 权威迁移到:(a) online_decision_log.jsonl 的 completion 决策行(每
    # request 恰一条,kind_counts 校验;tick = KV 生命周期终止时刻)、
    # (b) cpp.log phase-5 total_kv_actions(每 request ≥ 1 动作的计数级
    # 保证)。
    kv_missing_bad = []
    kv_tick_bad = []
    kv_order_bad = []
    kv_instance_bad = []
    if decision_log is not None and decision_rows:
        comp = decision_log["completion"]
        for request_id in manifest_ids:
            row = comp.get(request_id)
            if row is None:
                kv_missing_bad.append(request_id)
                continue
            completed_tick = (python_ledger.get(request_id, {})
                              .get("completed_unreconciled", {})
                              .get("completed_tick"))
            if row["tick"] != completed_tick:
                kv_tick_bad.append({
                    "request_id": request_id,
                    "completion_decision_tick": row["tick"],
                    "completed_tick": completed_tick})
            prefill = decision_log["prefill"].get(request_id)
            decode = decision_log["decode"].get(request_id)
            if prefill and decode and not (
                    prefill["tick"] <= decode["tick"] <= row["tick"]):
                kv_order_bad.append({
                    "request_id": request_id,
                    "prefill_tick": prefill["tick"],
                    "decode_tick": decode["tick"],
                    "completion_tick": row["tick"]})
            instance = ((row.get("decision") or {})
                        .get("kv_instance_after_completion"))
            if instance is not None and not (0 <= instance <= 53):
                kv_instance_bad.append({
                    "request_id": request_id,
                    "kv_instance_after_completion": instance})
        kv_low = []
        if cpp_audit is not None and cpp_audit["total_kv_actions"] \
                < len(manifest_ids):
            kv_low.append(cpp_audit["total_kv_actions"])
        active_left = []
        if cpp_counters is not None and cpp_counters["active"] != 0:
            active_left.append(cpp_counters["active"])
        check("R3a", not kv_missing_bad and not kv_low,
              "KV 生命周期缺失(每 request 恰一 completion 决策;C++ "
              "total_kv_actions >= 请求数): 缺 completion {} 条 / 动作"
              "计数过低 {}".format(len(kv_missing_bad), kv_low))
        check("R3b", not kv_tick_bad,
              "completion 决策 tick != ledger completed_tick: {} 条".format(
                  len(kv_tick_bad)))
        check("R3c", not kv_order_bad and not kv_instance_bad
              and not active_left,
              "KV 顺序/实例/收尾违例: 顺序 {} 条 / 实例越界 {} 条 / "
              "active 残留 {}".format(
                  len(kv_order_bad), len(kv_instance_bad), active_left))
    else:
        check("R3a", False, "online_decision_log.jsonl 缺失/为空")
        kv_low, active_left = [], []

    # ---- R4 排队账本对平 ----
    admitted_left = {
        rid: layers.get("admitted") for rid, layers in python_ledger.items()
        if layers.get("admitted") is not None}
    committed_left = {
        rid: layers.get("committed") for rid, layers in python_ledger.items()
        if layers.get("committed") is not None}
    check("R4a", not admitted_left,
          "run-end admitted layer not empty: {!r}".format(
              sorted(admitted_left)))
    check("R4b", not committed_left,
          "run-end committed layer not empty: {!r}".format(
              sorted(committed_left)))
    final_query = query_log[-1] if query_log else {}
    final_queued = (final_query.get("admitted_not_injected_queued") or {})
    check("R4c", final_queued.get("admitted_count") == 0,
          "final decision boundary admitted_count = {} (expected 0)".format(
              final_queued.get("admitted_count")))

    # ---- R5 injected-unfinished 对平 ----
    if query_log:
        last_row = query_log[-1]
        residuals = [
            entry for entry in last_row.get("injected_unfinished", [])
            if entry.get("node_count", 0) != 0]
        # 末决策边界残差 = 最后请求 decode 的 end barrier 控制节点(1 字节
        # all_reduce,无 compute、无时长模型;在最终 watch 命中时仍在途,
        # 于运行结束前完成——C++ 干净退出 active=0 即证)。因此 R5a 断言:
        # 残差非空(终请求 barrier 必然在途)且全部为 barrier 签名。
        residual_bad = []
        residual_groups = []
        for entry in residuals:
            for group in entry.get("per_request", []):
                residual_groups.append(group)
        for group in residual_groups:
            signature_ok = (
                group.get("stage") == "decode"
                and group.get("node_count") == 1
                and group.get("compute_ops") == 0
                and group.get("estimated_remaining_ns") == 0
                and group.get("comm_bytes") == 1)
            if not signature_ok:
                residual_bad.append(group)
        if not residual_groups:
            check("R5a", False,
                  "final epoch injected-unfinished fully drained (expected "
                  "end-barrier residual of the final request)")
        else:
            check("R5a", not residual_bad,
                  "{} non-barrier residual entries at final epoch: {!r}".format(
                      len(residual_bad), residual_bad[:5]))
        # 注入账本上限对平:任一边界 injected (request, stage) 节点总数
        # (跨 rank/代求和)不得超过该 (request, stage) 的 committed 节点数
        # (以 completed_nodes 逐对计数为上限,phase-7 契约;提交后逐批
        # 流入,完成时归零;整体上限 = 已提交总数)。
        over_committed = []
        missing_committed = []
        peak_by_key = {}
        for row in query_log:
            epoch_totals = {}
            for rank_entry in row.get("injected_unfinished", []):
                for group in rank_entry.get("per_request", []):
                    key = (group["request_id"], group["stage"])
                    epoch_totals[key] = (epoch_totals.get(key, 0)
                                         + group["node_count"])
            for key, total in epoch_totals.items():
                peak_by_key[key] = max(peak_by_key.get(key, 0), total)
        for (request_id, stage), peak in sorted(peak_by_key.items()):
            committed = committed_counts.get((request_id, stage), 0)
            if committed == 0:
                missing_committed.append({
                    "request_id": request_id, "stage": stage,
                    "peak_injected": peak})
            elif peak > committed:
                over_committed.append({
                    "request_id": request_id, "stage": stage,
                    "peak_injected": peak, "committed_nodes": committed})
        check("R5b", not over_committed and not missing_committed,
              "{} (request, stage) injected peak > committed node count;"
              "{} 注入对无 committed 记录".format(
                  len(over_committed), len(missing_committed)))
    else:
        check("R5a", False, "sensing_query_log.jsonl empty/absent")

    # ---- R6 ready 层对平(阶段 7 §10.1) ----
    # ready = 已准入且所在实例就绪(非忙)可服务的排队成员;断言每边界
    # ready ⊆ admitted(键集)且 ready ∩ issued == ∅(就绪成员尚未发射);
    # 末边界 ready 清空。
    ready_bad = []
    for row in query_log:
        ready = row.get("ready") or {}
        ready_ids = [entry["request_id"]
                     for entry in ready.get("detail", [])]
        admitted_ids = {
            entry["request_id"]
            for entry in ((row.get("admitted_not_injected_queued") or {})
                          .get("detail", []))
        }
        issued_ids = {
            entry["request_id"]
            for entry in (row.get("issued") or {}).get("detail", [])
        }
        outside = set(ready_ids) - admitted_ids
        overlap = set(ready_ids) & issued_ids
        if outside or overlap:
            ready_bad.append({
                "delivery_sequence": row["delivery_sequence"],
                "ready_outside_admitted": sorted(outside)[:5],
                "ready_and_issued_overlap": sorted(overlap)[:5],
            })
    check("R6a", not ready_bad,
          "{} boundary ready-layer violations (ready ⊆ admitted;"
          "ready ∩ issued == ∅)".format(len(ready_bad)))
    final_ready = ((query_log[-1] if query_log else {}).get("ready") or {})
    check("R6b", final_ready.get("ready_count", 0) == 0,
          "final decision boundary ready_count = {} (expected 0)".format(
              final_ready.get("ready_count", 0)))

    # ---- R7 issued 层对平(阶段 7 §10.1,核心) ----
    # 每边界 Python issued 键集(已发射未完成)⊆ C++ injected-unfinished
    # per_request 键集:Python 在 emit 时登记、completion 核销在策略完成
    # 处理前(基类 _settle_completions);C++ 摘要拍于同 delivery 的 tick-end
    # (watch 命中时已 finish 阶段节点、本批节点尚未 commit),两侧时刻对齐。
    # 反向残差 only_cpp 允许为"end-barrier 控制尾节点"已知窗口:watch 命中
    # 即报告阶段完成(Python 已核销 issued),但每 rank 的 1 字节 all_reduce
    # 控制节点仍在途(阶段 3 R5 同源语义)——断言残差全部为 barrier 签名
    # (单节点、无 compute、无时长);末边界 issued 清空。
    issued_only_python = []
    issued_residual_bad = []
    residual_peak = 0
    for row in query_log:
        py_issued = {
            entry["request_id"]
            for entry in (row.get("issued") or {}).get("detail", [])
        }
        cc_keys = set()
        cc_groups = []
        for rank_entry in row.get("injected_unfinished", []):
            for group in rank_entry.get("per_request", []):
                cc_keys.add(group["request_id"])
                cc_groups.append(group)
        only_python = py_issued - cc_keys
        only_cpp = cc_keys - py_issued
        residual_peak = max(residual_peak, len(only_cpp))
        if only_python:
            issued_only_python.append({
                "delivery_sequence": row["delivery_sequence"],
                "only_python": sorted(only_python)[:5]})
        # 残差核对:only_cpp 的每个组必须是 barrier 签名(控制尾节点)。
        for group in cc_groups:
            if group["request_id"] not in only_cpp:
                continue
            signature_ok = (
                group.get("node_count", 0) == 1
                and group.get("compute_ops", 0) == 0
                and group.get("estimated_remaining_ns", 0) == 0)
            if not signature_ok:
                issued_residual_bad.append({
                    "delivery_sequence": row["delivery_sequence"],
                    "request_id": group["request_id"],
                    "stage": group.get("stage"),
                    "node_count": group.get("node_count"),
                    "compute_ops": group.get("compute_ops"),
                    "comm_bytes": group.get("comm_bytes"),
                    "estimated_remaining_ns":
                        group.get("estimated_remaining_ns"),
                })
    check("R7a1", not issued_only_python,
          "{} boundaries with Python-issued outside C++ injected (Python "
          "issued ⊆ C++ keys required)".format(len(issued_only_python)))
    check("R7a2", not issued_residual_bad,
          "{} non-barrier residual groups in the only-C++ window (end-"
          "barrier control tails only)".format(len(issued_residual_bad)))
    final_issued = ((query_log[-1] if query_log else {}).get("issued") or {})
    check("R7b", final_issued.get("issued_count", 0) == 0,
          "final decision boundary issued_count = {} (expected 0)".format(
              final_issued.get("issued_count", 0)))

    # ---- R8 network pending/active 计数对账(阶段 7 §10.1) ----
    # C++ injected-unfinished 摘要内部一致性:每 rank 未完成节点数 >=
    # 已发射在飞节点数 + 空闲未发射节点数(剩余 = 依赖未满足的未发射
    # 中间节点,段内依赖链正常形态);每在飞 request 至少 1 个在飞节点
    # (跨 rank 求和);峰值统计供报告列示。
    cc_internal_bad = []
    cc_cover_bad = []
    peak_node = peak_in_flight = peak_free = 0
    for row in query_log:
        node_total = in_flight_total = free_total = 0
        for rank_entry in row.get("injected_unfinished", []):
            node_total += rank_entry.get("node_count", 0)
            in_flight_total += rank_entry.get("in_flight_node_count", 0)
            free_total += rank_entry.get("free_node_count", 0)
            if (rank_entry.get("node_count", 0)
                    < rank_entry.get("in_flight_node_count", 0)
                    + rank_entry.get("free_node_count", 0)):
                cc_internal_bad.append({
                    "delivery_sequence": row["delivery_sequence"],
                    "rank": rank_entry.get("rank"),
                    "node_count": rank_entry.get("node_count"),
                    "in_flight": rank_entry.get("in_flight_node_count"),
                    "free": rank_entry.get("free_node_count"),
                })
        peak_node = max(peak_node, node_total)
        peak_in_flight = max(peak_in_flight, in_flight_total)
        peak_free = max(peak_free, free_total)
        issued_count = (row.get("issued") or {}).get("issued_count", 0)
        if issued_count and in_flight_total < issued_count:
            cc_cover_bad.append({
                "delivery_sequence": row["delivery_sequence"],
                "issued_count": issued_count,
                "in_flight_nodes": in_flight_total,
            })
    check("R8a", not cc_internal_bad,
          "{} boundary summary inconsistencies (node_count >= in_flight + "
          "free)".format(len(cc_internal_bad)))
    check("R8b", not cc_cover_bad,
          "{} boundaries where in-flight nodes < issued requests".format(
              len(cc_cover_bad)))

    # ---- 汇总 ----
    balanced = not failures
    assignment_count = (cpp_audit["total_assignments"]
                        if cpp_audit is not None else 0)
    kv_count = (cpp_audit["total_kv_actions"]
                if cpp_audit is not None else 0)
    nodes_delta = None
    if cpp_audit is not None:
        nodes_delta = cpp_audit["total_nodes"] - completed_nodes_total
    if args.report:
        with open(args.report, "w", encoding="utf-8") as out:
            out.write(_render_report(
                balanced=balanced, failures=failures, manifest_count=len(
                    manifest_ids),
                arrival_epochs=arrival_epochs,
                delivery_epochs=len(query_log),
                assignment_count=assignment_count,
                kv_count=kv_count,
                cpp_counters=cpp_counters,
                cpp_audit=cpp_audit,
                completed_nodes_total=completed_nodes_total,
                nodes_delta=nodes_delta,
                lifecycle_bad=lifecycle_bad,
                watch_total_bad=watch_total_bad,
                batch_watch_bad=batch_watch_bad,
                batch_rank_bad=batch_rank_bad,
                node_total_bad=node_total_bad,
                audit_bad=audit_bad,
                kv_missing_bad=kv_missing_bad, kv_tick_bad=kv_tick_bad,
                kv_order_bad=kv_order_bad, kv_instance_bad=kv_instance_bad,
                kv_low=kv_low, active_left=active_left,
                admitted_left=admitted_left, committed_left=committed_left,
                ready_bad=ready_bad, final_ready=final_ready,
                issued_only_python=issued_only_python,
                issued_residual_bad=issued_residual_bad,
                residual_peak=residual_peak, final_issued=final_issued,
                cc_internal_bad=cc_internal_bad, cc_cover_bad=cc_cover_bad,
                peak_node=peak_node, peak_in_flight=peak_in_flight,
                peak_free=peak_free))
        print("[ledger_reconcile] report written: {}".format(args.report))
    print("[ledger_reconcile] verdict: {}".format(
        "对平 (balanced)" if balanced else "不平 (unbalanced)"))
    if failures:
        for item, evidence in failures:
            print("[ledger_reconcile] FAIL: {}: {}".format(item, evidence))
    return 0 if balanced else 1


def _render_report(balanced, failures, manifest_count, arrival_epochs,
                   delivery_epochs, assignment_count, kv_count, cpp_counters,
                   cpp_audit, completed_nodes_total, nodes_delta,
                   lifecycle_bad, watch_total_bad, batch_watch_bad,
                   batch_rank_bad, node_total_bad, audit_bad,
                   kv_missing_bad, kv_tick_bad, kv_order_bad,
                   kv_instance_bad, kv_low, active_left,
                   admitted_left, committed_left,
                   ready_bad, final_ready, issued_only_python,
                   issued_residual_bad, residual_peak, final_issued,
                   cc_internal_bad, cc_cover_bad,
                   peak_node, peak_in_flight, peak_free):
    lines = []
    lines.append("# 结束总账核对报告 — 分层账本逐层对平(阶段 7 §10.1,2026-08-15)")
    lines.append("")
    lines.append("> 方案 §6.2 操作 3 / contract ⑥ + §10.1:Python KV/排队账本 vs C++")
    lines.append("> 执行事实,逐 request/stage 对平;八层 remaining-load 账本")
    lines.append("> 逐层对账(R0-R8)。输入:strategy 感知模式运行 — 数据源为")
    lines.append("> 归档 jsonl(request completed_nodes / graph_batch_digests /")
    lines.append("> online_decision_log)+ cpp.log phase-5/4 审计计数器;")
    lines.append("> response_*.json 为消费即删,不再作输入。")
    lines.append("")
    lines.append("## 判定")
    lines.append("")
    lines.append("**{}**".format("对平(balanced)" if balanced else
                                "不平(unbalanced)"))
    lines.append("")
    lines.append("| 对平项 | 结果 |")
    lines.append("|---|---|")
    lines.append("| R0 集合对平(manifest / Python completed-unreconciled / C++ REQUEST_COMPLETE / cpp.log completed = {};prefill/decode 完成事实各 {};零重复) | {} |".format(
        manifest_count, manifest_count,
        "PASS" if not failures else "FAIL"))
    lines.append("| R1 逐 request 生命周期对平(admitted<=commit<=prefill<=decode==complete;gen 0/1/1) | {} |".format(
        "PASS" if not lifecycle_bad else "FAIL: {} 条".format(len(lifecycle_bad))))
    lines.append("| R2 committed 节点对平(phase-7:每 (request, stage) 恰注册一次 watch;completed_nodes 对 == ledger 对 == digest watch 和 == cpp total_watches;批级摘要 watch_count∈[0,len(ranks)] 且与 ranks 空性一致;rank ⊆ [0,53];digest node 和 == cpp total_nodes;phase-4 审计全零) | {} |".format(
        "PASS" if not (watch_total_bad or batch_watch_bad or batch_rank_bad
                       or node_total_bad or audit_bad)
        else "FAIL: watch 总数 {} 条 / 批 watch {} 条 / 批 rank {} 条 / node 总数 {} 条 / 审计 {} 条".format(
            len(watch_total_bad), len(batch_watch_bad), len(batch_rank_bad),
            len(node_total_bad), len(audit_bad))))
    lines.append("| R3 KV 账本对平(decision_log 每 request 恰一次 completion 决策;completion tick == ledger completed_tick;prefill<=decode<=completion;kv 实例 id ∈ [0,53];cpp total_kv_actions >= 请求数;运行结束 active=0) | {} |".format(
        "PASS" if not (kv_missing_bad or kv_tick_bad or kv_order_bad
                       or kv_instance_bad or kv_low or active_left)
        else "FAIL: 缺 completion={} tick={} 乱序={} 实例={} kv_low={} active_left={}".format(
            len(kv_missing_bad), len(kv_tick_bad), len(kv_order_bad),
            len(kv_instance_bad), kv_low, active_left)))
    lines.append("| R4 排队账本对平(运行结束 admitted/committed 清空;末边界 admitted_count=0) | {} |".format(
        "PASS" if not (admitted_left or committed_left) else
        "FAIL: admitted={} committed={}".format(
            len(admitted_left), len(committed_left))))
    lines.append("| R5 injected-unfinished 对平(末决策边界残差 = 终请求 decode end-barrier 控制节点;注入峰值 ≤ committed 节点数) | {} |".format(
        "PASS" if balanced else "FAIL"))
    lines.append("| R6 ready 层对平(§10.1:每边界 ready ⊆ admitted;ready ∩ issued == ∅;末边界 ready=0) | {} |".format(
        "PASS" if not ready_bad and final_ready.get("ready_count", 0) == 0
        else "FAIL: 边界违例 {} 条 / 末边界 ready={}".format(
            len(ready_bad), final_ready.get("ready_count", 0))))
    lines.append("| R7 issued 层对平(§10.1:每边界 Python issued ⊆ C++ injected 键集;only-C++ 残差全部为 end-barrier 控制尾节点签名;末边界 issued=0) | {} |".format(
        "PASS" if not issued_only_python and not issued_residual_bad
        and final_issued.get("issued_count", 0) == 0
        else "FAIL: only-python {} 条 / 残差签名 {} 条 / 末边界 issued={} / 残差峰值 {}".format(
            len(issued_only_python), len(issued_residual_bad),
            final_issued.get("issued_count", 0), residual_peak)))
    lines.append("| R8 network pending/active 计数对账(§10.1:摘要内部一致 node_count ≥ in_flight+free;在飞节点数 ≥ issued 数) | {} |".format(
        "PASS" if not cc_internal_bad and not cc_cover_bad
        else "FAIL: 内部不一致 {} 条 / 覆盖不足 {} 条".format(
            len(cc_internal_bad), len(cc_cover_bad))))
    lines.append("")
    lines.append("## 运行规模")
    lines.append("")
    lines.append("- manifest requests: {};arrival epochs: {};delivery epochs: {};assignments(cpp phase-5): {};kv_actions(cpp phase-5): {};cpp.log counters: {}".format(
        manifest_count, arrival_epochs, delivery_epochs, assignment_count,
        kv_count, cpp_counters if cpp_counters else "n/a"))
    if cpp_audit is not None:
        lines.append("- phase-5 批级审计: total_nodes={} / total_watches={} / graph_batch_count={} / single_node_bridge_count={}".format(
            cpp_audit["total_nodes"], cpp_audit["total_watches"],
            cpp_audit["graph_batch_count"],
            cpp_audit["single_node_bridge_count"]))
        lines.append("- committed 明细: completed_nodes 条数 = {};与 cpp total_nodes 之差 = {}(有界残差,详见 R2a 说明)".format(
            completed_nodes_total,
            "0" if nodes_delta == 0 else "{}".format(nodes_delta)))
    lines.append("- network pending/active 峰值(§10.1,C++ 摘要):未完成节点 {} / 在飞 {} / 空闲未发射 {}".format(
        peak_node, peak_in_flight, peak_free))
    lines.append("- R7 only-C++ 残差峰值(§10.1,end-barrier 控制尾节点在途窗口): {}".format(
        residual_peak))
    lines.append("")
    lines.append("## 八层 remaining-load 账本口径(§10.1,显式保留)")
    lines.append("")
    lines.append("| 层 | 账本归属 | 状态 |")
    lines.append("|---|---|---|")
    lines.append("| admitted(Python 排队账本成员) | Python 常驻 | 对平 R4 |")
    lines.append("| committed(GraphBatch 已提交) | Python 常驻 | 对平 R2 |")
    lines.append("| ready(就绪可服务) | Python 边界视图 | 对平 R6 |")
    lines.append("| issued(已发射未完成) | Python 常驻 | 对平 R7 |")
    lines.append("| remote-memory FIFO | 不适用(wscllm 无远端内存后端 FIFO) | 显式占位,无账本 |")
    lines.append("| network pending/active | C++ 执行事实(injected-unfinished 摘要) | 对平 R8(计数/审计) |")
    lines.append("| local HBM job | 无 local-HBM job 账本(计时模型已激活:"
                 "comm 边端点计费 hbm_charge 在案,裁决 #35;缺的是独立 "
                 "local-HBM 节点账本) | 显式占位,无账本 |")
    lines.append("| completed-unreconciled | Python 常驻 | 对平 R0 |")
    lines.append("")
    if failures:
        lines.append("## 差异清单(逐项归因)")
        lines.append("")
        lines.append("| 对平项 | 证据 |")
        lines.append("|---|---|")
        for item, evidence in failures:
            lines.append("| {} | {} |".format(item, evidence))
    lines.append("")
    lines.append("## 复核方式")
    lines.append("")
    lines.append("`python3 online/verify/ledger_reconcile.py --bridge-dir "
                 "<run>/bridge --manifest <ET_DIR>/manifest.json --cpp-log "
                 "<run>/cpp.log --report <out>.md`;退出 0 = 对平。")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
