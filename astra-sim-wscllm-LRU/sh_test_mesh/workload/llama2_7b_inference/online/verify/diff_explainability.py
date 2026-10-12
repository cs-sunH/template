#!/usr/bin/env python3
"""diff_explainability.py -- 阶段 3 差异可解释性(方案 §6.2 操作 4)。

登记(py-verify-D2 2026-09-27):手动 CLI 审计门工具——本仓无自动流水线
入口消费其退出码,审计门判定依赖人工运行并核对退出 0/1/2(见文件尾
"复核方式");不被 pytest 收集,属有意保留的手动对拍装置。

strategy 感知开 vs 关感知 对比,逐决策归类:

  无差异            -- 决策行(seq/tick/kind/request_id/decision)逐字节一致
  真实完成事件时序   -- 同一 (request, stage) 的 C++ 完成事实 tick 不同
                       (物理完成事件时序差异),导致后续决策 tick/内容不同
  排队状态差异       -- 决策内容不同且不由完成事件时序解释(决策输入——
                       排队深度/KV 状态——不同;感知泄漏红线,E1 修复
                       2026-09-27 起独立判 DEFECT)

每类差异必须给出解释;无法归类的差异 = 缺陷,报告并退出 1。
排队状态差异(决策输入不同 = 感知数据泄漏进策略判据)即缺陷,
同样退出 1——不得以"已归类"为由放行(E1 修复:恢复基线口径
"任何真实差异都判验证失败";此前该类仅 DIFF 级恒 exit 0)。

输入(两个 strategy 运行目录):
  --baseline-run <dir>   关感知运行(默认策略运行)
  --sensing-run <dir>    感知开运行
  --report <path.md>     差异报告(markdown)

比较对象(缺陷 3 修复,2026-08-16:原 kv_actions/assignments 维度从
response_*.json 收集,phase-7 §10.3 起 response 消费即删,两侧恒读成
空列表,空==空打出假绿 PASS(签名 sha256=4f53cda1…,即空数组 [] 的
哈希)。数据源迁移到归档 jsonl,与 ledger_reconcile 054118e 同思路):
  - online_decision_log.jsonl(逐行,seq 对齐)
  - graph_batch_digests.jsonl(逐行;生产缺省不产出,env SH_GRAPH_DIGESTS=1
    开启——两侧同缺时本维度与下一维度标 SKIP,不阻断审计门)
  - online_decision_log.jsonl 的 KV 决策载荷流(每行 decision 字段;
    tick/seq 无关的 KV 策略内容投影——原 response kv_actions 维度的
    归档替代)
  - graph_batch_digests.jsonl 的批级 assignment 摘要流(delivery_sequence/
    reasons/ranks/node_count/edge_count/watch_count/content_sha256,
    content_sha256 即批内 nodes+parent_edges 内容摘要——原 response
    assignments 维度的归档替代)
  - request_*.json 的 reasons / arrivals / completed_groups
    (排除 ledger_summary 字段——感知数据载体,差异属设计内,预期差异)
  - cpp.log 门计数器(delivery_count / event_count / completed / accepted)

jsonl 位置解析:bridge/ 优先,次选 results/(运行归档布局:官方 runner
结束后把 jsonl 从 bridge/ 移入 results/;两种布局均支持,内容一致)。

graph_batch_digests.jsonl 生产缺省不产出(env SH_GRAPH_DIGESTS=1 开启,
官方 runner 不开):两侧同缺时 digest 相关维度(批级摘要逐行对平/批级
assignment 摘要流)标显式 SKIP、退出码非 2——对齐 ledger_reconcile 同
数据源缺失的 SKIP 降级语义,审计门对默认 run 可用;单侧缺一侧有 = 运行
配置不对齐(SH_GRAPH_DIGESTS 一侧开一侧关),批级维度不可比,判 DEFECT。
decision log 为核心输入:缺失/为空照旧 fail-closed。

fail-closed 契约(空数据源禁止比较,禁止空==空真空通过):决策行缺
decision 载荷、bridge 无 request_*.json、cpp.log 缺失或未解析出任何门
计数器——立即报错退出 2,不产出报告。

退出码:0 = 全部差异已归类并解释且无排队状态差异(预期:全部无差异);
1 = 存在无法归类的差异、行数对不上或排队状态差异(感知泄漏红线);
2 = 核心数据源缺失/为空(fail-closed)。"""

import argparse
import gzip
import hashlib
import json
import os
import re
import sys

_ONLINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ONLINE_DIR not in sys.path:
    sys.path.insert(0, _ONLINE_DIR)
from bridge_request_journal import iter_request_records  # noqa: E402


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------

def _fail_closed(message):
    """空/缺数据源 fail-closed(缺陷 3 修复):立即退出 2,禁止真空比较。"""
    print("[diff_explainability] FAIL-CLOSED: {}".format(message),
          file=sys.stderr)
    raise SystemExit(2)


def _iter_jsonl(path):
    # errors="replace":非 UTF-8 字节裸崩 UnicodeDecodeError 会吞掉全部
    # 后续记账/报告(M-34 同款,对齐本文件 cpp.log 通道的宽松读纪律)。
    with open(path, "r", encoding="utf-8", errors="replace") as source:
        for line in source:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_jsonl_list(path):
    return list(_iter_jsonl(path))


def _resolve_jsonl(run_dir, name):
    """归档 jsonl 位置解析:bridge/ 优先,次选 results/(runner 归档布局)。
    两者皆缺 = 数据源缺失,fail-closed(不回退 response_*.json——该文件
    phase-7 §10.3 起消费即删,按旧契约读只会得到空集,即缺陷 3 的假绿)。"""
    path = _resolve_jsonl_or_none(run_dir, name)
    if path is None:
        _fail_closed("归档数据源缺失:{} 不存在于 {} 与 {}(空数据源禁止比较;"
                     "response_*.json 消费即删,不可作为数据源)".format(
                         name,
                         os.path.join(run_dir, "bridge", name),
                         os.path.join(run_dir, "results", name)))
    return path


def _resolve_jsonl_or_none(run_dir, name):
    """同 _resolve_jsonl 的解析顺序,缺失返回 None(不 fail-closed)——
    供生产缺省不产出的审计产物(graph_batch_digests.jsonl)走 SKIP 降级。"""
    bridge_path = os.path.join(run_dir, "bridge", name)
    results_path = os.path.join(run_dir, "results", name)
    if os.path.exists(bridge_path):
        return bridge_path
    if os.path.exists(results_path):
        return results_path
    return None



def _load_run(run_dir):
    bridge_dir = os.path.join(run_dir, "bridge")
    if not os.path.isdir(bridge_dir):
        _fail_closed("run 目录缺少 bridge/:{}".format(run_dir))
    decision_log_path = _resolve_jsonl(run_dir, "online_decision_log.jsonl")
    decision_log = _load_jsonl_list(decision_log_path)
    if not decision_log:
        _fail_closed("online_decision_log.jsonl 为空(0 行):{}——空数据源"
                     "禁止比较".format(decision_log_path))
    # graph_batch_digests.jsonl 生产缺省不产出(SH_GRAPH_DIGESTS 缺省 0,
    # 官方 runner 不开):文件缺失不再 fail-closed 退出 2(对任何默认 run
    # 恒退 2 = 审计门不可用),对齐 ledger_reconcile 的 SKIP 降级——两侧
    # 同缺由调用方对 digest 相关维度出显式 SKIP 行;文件在场但 0 行仍是
    # 空数据源,照旧 fail-closed(禁止空==空真空通过)。
    digests_path = _resolve_jsonl_or_none(run_dir, "graph_batch_digests.jsonl")
    digests = _load_jsonl_list(digests_path) if digests_path else []
    if digests_path and not digests:
        _fail_closed("graph_batch_digests.jsonl 为空(0 行):{}——空数据源"
                     "禁止比较".format(digests_path))

    # ---- KV 决策载荷流(原 response kv_actions 维度的归档替代)----
    # 每 decision 行投影为 {kind, request_id, decision}(tick/seq/priority
    # 属时序维度,已由决策行逐行对平覆盖;此处隔离 KV 策略内容本身)。
    # 载荷字段名各仓调度器不同(face/wscllm 的 history_action/prefill_
    # assignment_key、sh_2.0 的 *_eviction_count、sh_3.0 的 kv_instance_
    # after_completion 等),故按载荷整体做稳定摘要,不耦合字段名。
    kv_hasher = hashlib.sha256()
    kv_rows = 0
    for row in decision_log:
        decision = row.get("decision")
        if not isinstance(decision, dict) or not decision:
            _fail_closed("决策行缺 KV 决策载荷(decision 缺失/为空):"
                         "seq={} request_id={}({})".format(
                             row.get("seq"), row.get("request_id"),
                             decision_log_path))
        kv_hasher.update(json.dumps(
            {"kind": row.get("kind"), "request_id": row.get("request_id"),
             "decision": decision},
            sort_keys=True, separators=(",", ":")).encode("utf-8"))
        kv_rows += 1
    kv_digest = kv_hasher.hexdigest()

    # ---- 批级 assignment 摘要流(原 response assignments 维度的归档替代)----
    # content_sha256 为批内 nodes+parent_edges 内容摘要(含 assignment
    # 结构);ranks/node_count/edge_count/watch_count 为批级放置摘要。
    assignment_hasher = hashlib.sha256()
    assignment_rows = 0
    for row in digests:
        assignment_hasher.update(json.dumps(
            [row.get("delivery_sequence"), row.get("tick"),
             row.get("reasons"), row.get("ranks"), row.get("node_count"),
             row.get("edge_count"), row.get("watch_count"),
             row.get("content_sha256")],
            sort_keys=True, separators=(",", ":")).encode("utf-8"))
        assignment_rows += 1
    assignment_digest = assignment_hasher.hexdigest()

    # request_*.json:C++ 完成事实 + 到达事实(排除 ledger_summary 载体)。
    completions = {}   # request_id -> {"prefill": tick, "decode": tick,
                       #                "request_complete": tick}
    arrivals = []      # (tick, request_id) 序列
    reasons = 0
    request_digest_excl_summary = hashlib.sha256()
    request_record_count = 0
    for seq, req in iter_request_records(bridge_dir):
        request_record_count += 1
        arrivals.extend((req["tick"], record["request_id"])
                        for record in req.get("arrivals") or [])
        for group in req.get("completed_groups") or []:
            completions.setdefault(group["request_id"], {})[group["stage"]] = \
                req["tick"]
        for group in req.get("completed_groups") or []:
            if not group.get("stage"):
                completions[group["request_id"]]["request_complete"] = \
                    req["tick"]
        reasons += len(req.get("reasons") or [])
        _digest_request_excluding_summary(request_digest_excl_summary, req)
    if request_record_count == 0:
        _fail_closed(
            "bridge request journal/legacy request files are empty:{}——空数据源"
            "禁止比较".format(bridge_dir))
    request_digest = request_digest_excl_summary.hexdigest()

    cpp_log_path = os.path.join(run_dir, "cpp.log")
    counters = _load_cpp_counters(cpp_log_path)
    if counters is None:
        _fail_closed("cpp.log 缺失或未解析出任何门计数器:{}——空数据源"
                     "禁止比较".format(cpp_log_path))
    return {
        "decision_log": decision_log,
        "digests": digests,
        "kv_digest": kv_digest,
        "kv_rows": kv_rows,
        "assignment_digest": assignment_digest,
        "assignment_rows": assignment_rows,
        "completions": completions,
        "arrivals": arrivals,
        "reasons": reasons,
        "request_digest": request_digest,
        "counters": counters,
        "bridge_dir": bridge_dir,
    }


def _digest_request_excluding_summary(hasher, req):
    """对 request_*.json 做稳定序列化摘要,显式排除 ledger_summary
    (感知数据载体:关感知 = 空数组,感知开 = 内容;差异属设计内)。"""
    if "ledger_summary" in req:
        req = {key: value for key, value in req.items()
               if key != "ledger_summary"}
    hasher.update(json.dumps(req, sort_keys=True).encode("utf-8"))


def _load_cpp_counters(cpp_log_path):
    """cpp.log 门计数器解析。

    S5 修复(2026-09-29):C++ 实际输出(main_online.cc)为单行
    ``[online] gate counters: event_count=… delivery_count=…
    coalescing_ratio=… tick_end_without_decision_count=…``;旧正则
    ``\\[online\\] delivery_count=(\\d+)`` / ``\\[online\\] event_count=(\\d+)``
    对该行永不匹配,event/delivery 两计数从不进入比较,感知开/关差异被
    掩盖为假 PASS。现按行解析该行全部 key=value 对;gate counters 行缺失
    或 event_count/delivery_count 缺失 = 门计数器未解析出,返回 None 由
    调用方 fail-closed 退出 2(与既有空数据源契约一致)。"""
    if os.path.exists(cpp_log_path):
        opener = open
    elif os.path.exists(cpp_log_path + ".gz"):
        cpp_log_path += ".gz"
        opener = gzip.open
    else:
        return None
    with opener(cpp_log_path, "rt", encoding="utf-8", errors="replace") as source:
        text = source.read()
    counters = {}
    gate = re.search(r"^\[online\] gate counters:(.+)$", text, re.MULTILINE)
    if gate:
        for token in gate.group(1).split():
            key, sep, value = token.partition("=")
            if not sep or not value:
                continue
            try:
                counters[key] = int(value)
            except ValueError:
                try:
                    counters[key] = float(value)
                except ValueError:
                    continue
    if "event_count" not in counters or "delivery_count" not in counters:
        # fail-closed:门计数器行缺失/不完整,禁止空==空真空通过。
        return None
    service = re.search(r"\[online\] service counters: accepted=(\d+) "
                        r"completed=(\d+) active=(\d+) pending_alarm=(\d+)",
                        text)
    if service:
        counters.update({
            "accepted": int(service.group(1)),
            "completed": int(service.group(2)),
            "active": int(service.group(3)),
            "pending_alarm": int(service.group(4)),
        })
    return counters or None


# ---------------------------------------------------------------------------
# 逐决策归类
# ---------------------------------------------------------------------------

def classify_differences(baseline, sensing):
    """返回 (categories, differences):
      categories:  {类别: 计数}
      differences: [ {类别, seq, request_id, 说明, 证据} ]
    无差异行不产出差异条目;仅差异行进入分类。"""
    # 决策行数不一致(M31 修复)独立成类:此前折入"排队状态差异"后与
    # docstring 契约"行数对不上 → 退出 1"矛盾(DEFECT 级别全仓零产生点,
    # 审计器恒假绿)。归入本类的差异在 main 判定为 DEFECT。
    categories = {"无差异": 0, "真实完成事件时序": 0, "排队状态差异": 0,
                  "决策行数不一致": 0}
    differences = []
    bl = baseline["decision_log"]
    sn = sensing["decision_log"]
    max_len = max(len(bl), len(sn))
    for idx in range(max_len):
        if idx < len(bl) and idx < len(sn) and bl[idx] == sn[idx]:
            categories["无差异"] += 1
            continue
        if idx >= len(bl) or idx >= len(sn):
            row = bl[idx] if idx < len(bl) else sn[idx]
            request_id = row.get("request_id", "")
            differences.append({
                "类别": "决策行数不一致",
                "seq": idx,
                "request_id": request_id,
                "说明": "决策行数不一致(决策序列长度不同,感知开/关决策边界数"
                        "不同):baseline={} sensing={}".format(len(bl), len(sn)),
                "证据": json.dumps(row, sort_keys=True),
            })
            categories["决策行数不一致"] += 1
            continue
        baseline_row = bl[idx]
        sensing_row = sn[idx]
        request_id = (baseline_row.get("request_id")
                      or sensing_row.get("request_id") or "")
        # 归类 1:同一 request 的 C++ 完成事实 tick 是否不同。
        tick_diffs = _completion_tick_diffs(
            baseline["completions"], sensing["completions"], request_id)
        if tick_diffs:
            category = "真实完成事件时序"
        else:
            category = "排队状态差异"
        categories[category] += 1
        differing_keys = sorted(
            set(baseline_row) ^ set(sensing_row)
            | {key for key in baseline_row
               if baseline_row.get(key) != sensing_row.get(key)})
        differences.append({
            "类别": category,
            "seq": idx,
            "request_id": request_id,
            "说明": "决策行不同:keys={}".format(differing_keys),
            "证据": "baseline={} sensing={}".format(
                json.dumps(baseline_row, sort_keys=True),
                json.dumps(sensing_row, sort_keys=True)),
        })
    return categories, differences


def _completion_tick_diffs(bl_completions, sn_completions, request_id):
    if request_id not in bl_completions and request_id not in sn_completions:
        return []
    both = set(bl_completions.get(request_id, {})) | \
        set(sn_completions.get(request_id, {}))
    return [
        (stage, bl_completions.get(request_id, {}).get(stage),
         sn_completions.get(request_id, {}).get(stage))
        for stage in both
        if bl_completions.get(request_id, {}).get(stage) !=
        sn_completions.get(request_id, {}).get(stage)
    ]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="phase-3 sensing on/off difference explainability "
                    "(graph_batch_digests 两侧同缺时 digest 维度 SKIP,"
                    "核心数据源缺失 fail-closed 退出 2)")
    parser.add_argument("--baseline-run", required=True,
                        help="关感知 strategy 运行目录(含 bridge/ 与 cpp.log)")
    parser.add_argument("--sensing-run", required=True,
                        help="感知开 strategy 运行目录(含 bridge/ 与 cpp.log)")
    parser.add_argument("--report", default=None,
                        help="markdown 报告路径(缺省 stdout)")
    args = parser.parse_args(argv)

    baseline = _load_run(args.baseline_run)
    sensing = _load_run(args.sensing_run)

    findings = []  # (级别, 条目, 证据);级别: PASS / DIFF / DEFECT

    # ---- 1. 决策行逐行对平 ----
    categories, differences = classify_differences(baseline, sensing)
    row_mismatch = [diff for diff in differences
                    if diff["类别"] == "决策行数不一致"]
    if row_mismatch:
        # 契约(docstring 退出码):行数对不上 → 退出 1。M31 修复:DEFECT
        # 级别此前全仓零产生点,任意差异/行数不齐一律 exit 0(恒假绿)。
        findings.append(("DEFECT", "决策行数不一致(行数对不上)",
                         "baseline={} sensing={} 行,{} 条超界行(契约:行数"
                         "不齐 → 退出 1)".format(
                             len(baseline["decision_log"]),
                             len(sensing["decision_log"]),
                             len(row_mismatch))))
    elif differences:
        findings.append(("DIFF", "决策行差异",
                         "{} 条(见分类清单)".format(len(differences))))
    else:
        findings.append(("PASS", "决策行逐字节一致",
                         "{} 行 = 全部无差异".format(len(baseline["decision_log"]))))

    # 排队状态差异 = 感知泄漏红线(E1 修复 2026-09-27):决策输入(排队
    # 深度/KV 状态)不同即感知数据泄漏进策略判据,属不可归类解释的缺陷,
    # 独立判 DEFECT 退出 1——恢复基线口径"任何真实差异都判验证失败"
    # (此前该类仅 DIFF 级,机械归类后恒 exit 0)。
    queue_leaks = [diff for diff in differences
                   if diff["类别"] == "排队状态差异"]
    if queue_leaks:
        findings.append((
            "DEFECT", "排队状态差异(感知泄漏红线)",
            "{} 条决策输入不同且不由完成事件时序解释(首条 seq={} "
            "request_id={};契约:排队状态差异 → 退出 1)".format(
                len(queue_leaks), queue_leaks[0]["seq"],
                queue_leaks[0]["request_id"])))

    # 载体差异归类口径(M31 修复 + E1 倒置修正 2026-09-27):载体差异仅当
    # 决策行差异全部归因于"真实完成事件时序"(物理完成事实 tick 不同,
    # 载体差异为其下游表现)时降为 DIFF;决策行全同(载体差异无任何决策
    # 级归因)或差异含"排队状态差异"(决策输入本身不同,不可归因于完成
    # 时序)时,载体差异同样无法归类 → DEFECT(退出 1)。原
    # `DIFF if differences else DEFECT` 把决策行真实分歧机械降级为 DIFF,
    # 判定倒置:决策不同+载体不一致反而可能 exit 0,决策全同+载体不一致
    # 却 exit 1。
    timing_rows = [diff for diff in differences
                   if diff["类别"] == "真实完成事件时序"]
    carrier_level = "DIFF" if timing_rows and not queue_leaks else "DEFECT"

    # ---- 2. GraphBatch digest 逐行对平 ----
    # 生产缺省不产出(SH_GRAPH_DIGESTS 缺省 0,官方 runner 不开):两侧同
    # 缺 → digest 相关维度显式 SKIP(非 2 退出、不计失配,对齐
    # ledger_reconcile 同数据源缺失的 SKIP 降级);单侧缺一侧有 → 运行配置
    # 不对齐(SH_GRAPH_DIGESTS 一侧开一侧关),批级维度不可比,判 DEFECT。
    digests_both_missing = not baseline["digests"] and not sensing["digests"]
    digests_asymmetric = (not digests_both_missing
                          and (not baseline["digests"]
                               or not sensing["digests"]))
    if digests_both_missing:
        findings.append((
            "SKIP", "graph_batch_digests 逐行对平(数据源两侧同缺)",
            "graph_batch_digests.jsonl 默认不产出(env SH_GRAPH_DIGESTS=1 "
            "开启);批级摘要逐行对平与批级 assignment 摘要对平本报告未核,"
            "不计失配"))
    elif digests_asymmetric:
        findings.append((
            "DEFECT", "graph_batch_digests 单侧缺失",
            "baseline({} 行) sensing({} 行)——SH_GRAPH_DIGESTS 一侧开一侧"
            "关,运行配置不对齐,批级维度不可比".format(
                len(baseline["digests"]), len(sensing["digests"]))))
    elif baseline["digests"] == sensing["digests"]:
        findings.append(("PASS", "graph_batch_digests 逐行一致",
                         "{} 行".format(len(baseline["digests"]))))
    else:
        findings.append((carrier_level, "graph_batch_digests 不一致",
                         "baseline={} sensing={} 行".format(
                             len(baseline["digests"]), len(sensing["digests"]))))

    # ---- 3. KV 决策载荷 / 批级 assignment 摘要对平(归档 jsonl 数据源)----
    # 数据源:online_decision_log.jsonl decision 载荷流 / graph_batch_
    # digests.jsonl 批级摘要流(缺陷 3 修复:原 response_*.json 消费即删,
    # 两侧恒空→空==空假绿;现两侧行数进证据,空哈希签名 4f53cda1 不再可能)。
    if baseline["kv_digest"] == sensing["kv_digest"]:
        findings.append(("PASS", "KV 决策载荷一致(online_decision_log)",
                         "{} 行,sha256={}".format(
                             baseline["kv_rows"],
                             baseline["kv_digest"][:16])))
    else:
        findings.append((carrier_level, "KV 决策载荷不一致(online_decision_log)",
                         "baseline({} 行)={} sensing({} 行)={}".format(
                             baseline["kv_rows"], baseline["kv_digest"][:16],
                             sensing["kv_rows"],
                             sensing["kv_digest"][:16])))
    if digests_both_missing:
        findings.append((
            "SKIP", "批级 assignment 摘要对平(graph_batch_digests 两侧同缺)",
            "数据源默认不产出(env SH_GRAPH_DIGESTS=1 开启),同上 SKIP 行,"
            "不计失配"))
    elif digests_asymmetric:
        findings.append((
            "DEFECT", "批级 assignment 摘要对平不可比(graph_batch_digests "
            "单侧缺失)",
            "baseline({} 批) sensing({} 批)——见上方单侧缺失 DEFECT".format(
                len(baseline["digests"]), len(sensing["digests"]))))
    elif baseline["assignment_digest"] == sensing["assignment_digest"]:
        findings.append(("PASS", "批级 assignment 摘要一致(graph_batch_digests)",
                         "{} 批,sha256={}".format(
                             baseline["assignment_rows"],
                             baseline["assignment_digest"][:16])))
    else:
        findings.append((carrier_level, "批级 assignment 摘要不一致(graph_batch_digests)",
                         "baseline({} 批)={} sensing({} 批)={}".format(
                             baseline["assignment_rows"],
                             baseline["assignment_digest"][:16],
                             sensing["assignment_rows"],
                             sensing["assignment_digest"][:16])))

    # ---- 4. C++ 完成事实对平(排除 ledger_summary 载体) ----
    if baseline["completions"] == sensing["completions"]:
        findings.append(("PASS", "C++ 完成事实逐 request 一致",
                         "{} requests(prefill/decode/request_complete 三档"
                         " tick 全等)".format(len(baseline["completions"]))))
    else:
        diff_requests = sorted(
            set(baseline["completions"]) ^ set(sensing["completions"]))
        diff_ticks = [
            # S6 修复(2026-09-29):元组取值曾用 [request_id][stage] 硬下标,
            # 单侧缺失 stage(过滤用 .get(stage) 不拦)→ KeyError 裸崩;统一
            # 照 _completion_tick_diffs 的 .get 写法,缺失侧计 None 进清单。
            (request_id, stage,
             baseline["completions"][request_id].get(stage),
             sensing["completions"][request_id].get(stage))
            for request_id in sorted(set(baseline["completions"]) &
                                     set(sensing["completions"]))
            for stage in set(baseline["completions"][request_id]) |
            set(sensing["completions"][request_id])
            if baseline["completions"][request_id].get(stage) !=
            sensing["completions"][request_id].get(stage)
        ]
        findings.append((carrier_level, "C++ 完成事实差异",
                         "集合差={} tick差={} 条(首个:{})".format(
                             diff_requests[:5], len(diff_ticks),
                             diff_ticks[0] if diff_ticks else "-")))

    # ---- 5. 到达/原因 摘要对平 ----
    if baseline["arrivals"] == sensing["arrivals"]:
        findings.append(("PASS", "arrivals 一致", "{} 条".format(
            len(baseline["arrivals"]))))
    else:
        findings.append((carrier_level, "arrivals 不一致", "baseline={} sensing={}"
                         .format(len(baseline["arrivals"]),
                                 len(sensing["arrivals"]))))
    if baseline["reasons"] == sensing["reasons"]:
        findings.append(("PASS", "reasons 条数一致", "{} 条".format(
            baseline["reasons"])))
    else:
        findings.append((carrier_level, "reasons 条数不一致", "baseline={} sensing={}"
                         .format(baseline["reasons"], sensing["reasons"])))
    if baseline["request_digest"] == sensing["request_digest"]:
        findings.append(("PASS", "request_*.json 摘要一致(排除 "
                                 "ledger_summary 载体字段)", "sha256={}".format(
            baseline["request_digest"][:16])))
    else:
        findings.append((carrier_level, "request_*.json 摘要不一致(排除 "
                                        "ledger_summary)", "baseline={} sensing={}"
                         .format(baseline["request_digest"][:16],
                                 sensing["request_digest"][:16])))

    # ---- 6. cpp.log 门计数器对平 ----
    if baseline["counters"] == sensing["counters"]:
        findings.append(("PASS", "cpp.log 门计数器一致", "{}".format(
            baseline["counters"])))
    else:
        findings.append((carrier_level, "cpp.log 门计数器不一致", "baseline={} "
                                        "sensing={}".format(baseline["counters"],
                                                            sensing["counters"])))

    # ---- 判定 ----
    # DEFECT 优先(M31 修复):此前 `elif not defects` 恒真(defects 全仓零
    # 产生点)导致任意差异一律 exit 0;现在行数不齐与无法归类的载体差异
    # 都会产出 DEFECT 级 finding 并真实走到退出码 1。
    defects = [finding for finding in findings if finding[0] == "DEFECT"]
    diffs = [finding for finding in findings if finding[0] == "DIFF"]
    if defects:
        verdict = "存在无法归类的差异或行数不齐(DEFECT):{} 条".format(
            len(defects))
        ok = False
    elif not diffs:
        verdict = "全部无差异:感知开/关决策序列逐字节一致,差异已全部归类并解释"
        ok = True
    else:
        verdict = ("差异均已归类并解释(无缺陷差异):决策差异 {} 条,其余载体"
                   "差异均为设计内/事件时序类".format(len(differences)))
        ok = True

    _render(args.report, baseline, sensing, findings, categories,
            differences, verdict, ok)
    return 0 if ok else 1


def _render(report_path, baseline, sensing, findings, categories,
            differences, verdict, ok):
    lines = []
    lines.append("# 差异可解释性报告 — 感知开 vs 关感知(阶段 3)")
    lines.append("")
    lines.append("> 方案 §6.2 操作 4:strategy 感知开/关对比,逐决策归类;每类")
    lines.append("> 差异必须给出解释,无法归类 = 缺陷。输入:20.csv 前 30s ")
    lines.append("> 1177 requests / 112 sessions。")
    lines.append("> KV 决策载荷/批级 assignment 维度数据源 = 归档 jsonl")
    lines.append("> (缺陷 3 修复 2026-08-16:原 response_*.json 消费即删,")
    lines.append("> 空==空假绿;核心数据源 fail-closed 退出 2,")
    lines.append("> graph_batch_digests 两侧同缺时相关维度 SKIP 降级)。")
    lines.append("")
    lines.append("## 判定")
    lines.append("")
    lines.append("**{}**".format(verdict))
    lines.append("")
    lines.append("## 逐决策归类统计")
    lines.append("")
    lines.append("| 类别 | 条数 | 解释 |")
    lines.append("|---|---|---|")
    lines.append("| 无差异 | {} | 感知数据不进策略判据(红线 §0.4),决策输入"
                 "在两种模式下逐字节一致 |".format(categories.get("无差异", 0)))
    lines.append("| 真实完成事件时序 | {} | C++ 真实执行事实 tick 不同导致"
                 "后续决策 tick/内容不同(本报告预期 0:确定性运行) |".format(
                     categories.get("真实完成事件时序", 0)))
    lines.append("| 排队状态差异 | {} | 决策输入(排队深度/KV 状态)不同且不由"
                 "完成事件时序解释——感知泄漏红线,判 DEFECT 退出 1(E1 修复;"
                 "本报告预期 0) |".format(
                     categories.get("排队状态差异", 0)))
    lines.append("| 决策行数不一致 | {} | 决策序列长度不同(感知开/关决策"
                 "边界数不同)——契约:行数对不上 → 退出 1(DEFECT) |".format(
                     categories.get("决策行数不一致", 0)))
    lines.append("")
    lines.append("## 逐项比较")
    lines.append("")
    lines.append("| 比较对象 | 结果 | 证据 |")
    lines.append("|---|---|---|")
    for level, item, evidence in findings:
        lines.append("| {} | {} | {} |".format(level, item, evidence))
    lines.append("")
    lines.append("### 预期设计内差异(不进入逐决策分类)")
    lines.append("")
    lines.append("- request_*.json 的 `ledger_summary` 字段:感知数据载体,"
                 "关感知 = 空数组、感知开 = per-rank injected-unfinished "
                 "摘要;决策序列不受其影响(比较时已显式排除该字段)")
    lines.append("- cpp.log `[online] sensing: enabled` 日志行:仅感知运行"
                 "出现,日志级差异");
    lines.append("- 感知运行额外产出 ledger.jsonl / sensing_query_log.jsonl"
                 "(审计产物,非决策输入)")
    lines.append("")
    if differences:
        lines.append("## 差异清单(逐条归类)")
        lines.append("")
        lines.append("| 类别 | seq | request_id | 说明 | 证据 |")
        lines.append("|---|---|---|---|---|")
        for diff in differences:
            lines.append("| {} | {} | {} | {} | {} |".format(
                diff["类别"], diff["seq"], diff["request_id"],
                diff["说明"], diff["证据"]))
        lines.append("")
    lines.append("## 复核方式")
    lines.append("")
    lines.append("`python3 online/verify/diff_explainability.py "
                 "--baseline-run <off_run> --sensing-run <on_run> "
                 "--report <out>.md`;退出 0 = 全部差异已归类并解释且无排队"
                 "状态差异;退出 1 = 存在无法归类的差异、行数不齐或排队状态"
                 "差异(感知泄漏红线,DEFECT);"
                 "退出 2 = 核心数据源缺失/为空(fail-closed);"
                 "graph_batch_digests.jsonl 两侧同缺(默认不产出,env "
                 "SH_GRAPH_DIGESTS=1 开启)时 digest 相关维度标 SKIP,不影响"
                 "判定。")
    lines.append("")
    report_text = "\n".join(lines)
    if report_path:
        with open(report_path, "w", encoding="utf-8") as output:
            output.write(report_text)
        print("[diff_explainability] report written: {}".format(report_path))
    else:
        print(report_text)


if __name__ == "__main__":
    sys.exit(main())
