#!/usr/bin/env python3
"""ledger_reconcile.py -- 结束总账核对(方案 §6.2 操作 3 / contract ⑥;
阶段 7 §10.1 扩展为逐层对账)。

Python KV/排队账本 vs C++ 执行事实,逐 request/stage 对平。输入(同一
strategy 感知运行目录):

  --bridge-dir <run>/bridge   request_*.json(C++ 完成事实)
                              response_*.json(committed 节点 + kv_actions
                              + assignments)
                              graph_batch_digests.jsonl(批级 committed
                              摘要:watch_count / ranks / node_count;
                              可选——生产默认 SH_GRAPH_DIGESTS=0 不产出)
                              ledger.jsonl(Python 分层账本导出)
                              sensing_query_log.jsonl(决策边界两层剩余负载
                              查询快照)
  --manifest <ET_DIR>/manifest.json   全部请求事实(1177 requests)
  --config <trace_config.csv> 确定性重建分支的调度器配置(可选;缺省用
                              workload 默认 trace_config.csv;须与产生
                              本 run 的配置一致,重建前按 manifest 持久
                              字段 fail-closed 核对)
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
  R2 committed 节点对平(watch 成员账本):每 (request, stage) 恰注册一次
     watch,watch 成员数 == C++ completed_groups node_count(request 文件
     vs response 文件对平);watch 成员 rank ⊆ 该 (request, stage)
     committed 节点 rank 集。watch 成员 = 每 rank 最后一个该阶段节点
     (非全部阶段节点;阶段图可横跨 prefill+decode 两实例 rank),故
     committed rank 数可以大于成员数,不构成失配。
     R2b 加固(2026-08-16,wscllm 054118e 同款不变量;2026-09-29 M19 空性
     等价按列车化语义修正;# (2026-09-30, ported from the face repo) 上界
     改列车化口径):graph_batch_digests.jsonl 批级摘要合法——
     每批 watch_count ∈ [0, node_count](watch 按列车成员逐条注册,
     len(ranks) 上界结构性不成立)且 ranks 为空 ⇒ watch_count==0
     (单向空性约束;watch 锚点挂在本批节点上,零节点批不可能带 watch。
     旧双向等价 watch_count==0 ⇔ ranks 为空停留单请求时代——列车化后
     admission-only 批(graph_batch_builder.emit_admission_batch 只发
     arrival gate+readiness 屏障节点、无请求级 watch,test_graph_batch_
     builder.py::test_admission_emits_actions_without_body_or_watch 钉子)
     与纯逐出 mem 段合法地"有节点、零 watch",双向式对健康 run 必误报。
     watch_count 可为 2,PREFILL_DRAIN 同批 prefill+decode 双 watch /
     DECODE_COMPLETION+REQUEST_COMPLETE 可 3,禁 {0,1} 假设)、
     ranks ⊆ [0,--max-rank]
     (缺省 53 = 54-NPU 验收 mesh,跨硬件复用时改传)、
     sum(node_count) == C++ phase-5 total_nodes 且 sum(watch_count)
     == phase-5 total_watches;有 digest 才校验(2026-09-26 降级:
     生产默认 SH_GRAPH_DIGESTS=0 不产出本文件,数据源缺失时本项
     SKIP 而非 fail)。R2c:phase-4 end audit 全零(watch 生命
     周期干净收尾)。
  R3 KV 账本对平(Python KV 权威 vs C++ 完成事实):每 request 至少 1 条
     kv_action(trigger_request_id == R);"completion" phase 事件
     (retain_complete)的 planner_time_ns == completed_tick;所有 kv 事件
     tick <= completed_tick。
  R4 排队账本对平:运行结束 admitted/committed 层已清空(ledger.jsonl 无
     admitted/committed 条目);最后一决策边界 admitted_count == 0。
  R5 injected-unfinished 对平(sensing_query_log):末决策边界残差 = 终请求
     decode end-barrier 控制节点(列车迭代数字节的 all_reduce——拼 batch
     改造后 barrier 由每请求 1 字节改为每列车 iterations 字节;无 compute、
     无时长;最终 watch 命中时仍在途,运行结束前完成——C++ 干净退出
     active=0 即证),断言残差非空且全部为 barrier 签名;注入峰值(节点数,跨
     rank/代求和)≤ committed 节点数(提交后逐批流入,完成时归零)。
  R6 ready 层对平(§10.1):每边界 ready(就绪可服务排队成员,策略 override
     视图)⊆ admitted(键集)且 ready ∩ issued == ∅;末决策边界 ready==0。
  R7 issued 层对平(§10.1,核心):每边界 Python issued 键集(已发射未完成;
     emit 登记、completion 核销于策略完成处理前)⊆ C++ injected-unfinished
     per_request 键集(同 delivery 同 tick 摘要);only-C++ 残差全部为
     end-barrier 控制尾节点签名(watch 命中即报阶段完成、列车迭代数字节
     all_reduce 尾节点仍在途的已知窗口,与 R5 同源);末决策边界 issued==0。
  R8 network pending/active 计数对账(§10.1):C++ 摘要内部一致(每 rank
     node_count >= in_flight_node_count + free_node_count,剩余 = 依赖未
     满足的未发射中间节点);在飞节点数(跨 rank)>= issued request 数(每
     在飞 request 至少 1 节点);峰值统计列示。

八层账本口径(§10.1):admitted / committed / ready / issued / network
pending/active 参与对平;remote FIFO 与 local HBM 为"不适用"显式占位
(本仓远端内存后端与 LocalHbmBandwidthModel 均已启用——README §D/§G;
端口为并发流体计时、无 FIFO 等待账本,对账八层亦无 HBM 作业账本条目),
报告中列示不参与对平。

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
from online_scheduler_base import BATCH_TRAIN_PREFIX  # noqa: E402


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
            if request_id.startswith(BATCH_TRAIN_PREFIX):
                # 列车哨兵完成事实(T_max 截断列车/首步拆分唤醒,watch 以
                # train_id+prefill 注册)不属于任何请求——与调度器完成
                # 处理(face_online_scheduler 对同组按 BATCH_TRAIN_PREFIX
                # 路由核销)同款剥离,否则 R0c1/R0c3 把哨兵计入 request
                # 生命周期事实必然误报失配。
                continue
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


def _check_replay_config_against_manifest(config, manifest):
    """确定性重建 config ↔ run manifest fail-closed 核对。

    重建重放必须发生在与真实 run 相同的调度器配置下:manifest.json 顶部
    持久字段由 plan_materializer 物化时按当次 config 写入,这里逐项与本次
    载入 config 的派生值核对(kv_cache_policy / kv_reserve_context_tokens /
    request_queue_csv / hardware label——全部实际进入重放)。不一致即拒绝
    重建:静默重放会产生与 C++ 事实系统性失配的伪"不平"账本。manifest
    缺某键(旧 schema)时跳过该对,不做反向误报。"""
    pairs = [
        ("kv_management.policy",
         (manifest.get("kv_management") or {}).get("policy"),
         config.kv_cache_policy),
        ("kv_reserve_context_tokens",
         manifest.get("kv_reserve_context_tokens"),
         config.kv_reserve_context_tokens),
        ("request_queue_csv",
         manifest.get("request_queue_csv"),
         str(config.request_queue_csv)),
        ("hardware.label",
         (manifest.get("hardware") or {}).get("label"),
         config.hardware.label),
    ]
    mismatches = [
        "manifest {} = {!r} != config {!r}".format(key, recorded, actual)
        for key, recorded, actual in pairs
        if recorded is not None and recorded != actual
    ]
    if mismatches:
        raise SystemExit(
            "[ledger_reconcile] FAIL: 确定性重建配置与 run manifest 不一致"
            "(拒绝重建;用 --config 传入产生本 run 的 trace_config.csv):\n  "
            + "\n  ".join(mismatches))


def load_responses(bridge_dir, manifest_path=None, config_path=None):
    """从 response_*.json 收集:每 (request_id, stage) committed 节点的去重
    rank 集合与节点总数;全部 kv_actions;assignments 计数;注册过的 stage
    watch(members = {rank: node_id} 字典)。

    face 适配(阶段 7 §10.3 起 response 消费即删):bridge 无 response 文件
    时,改用确定性 Python 侧重建——request_*.json(C++ 实际交付的 delta
    序列)逐一喂给 FaceOnlineScheduler,重建每个 GraphBatch 的
    watches/kv_actions/assignments/节点(调度器为确定性函数,与在线运行
    交付一致;幂等 fixture 同机制验证)。需要 --manifest;配置经 --config
    传入(缺省 workload 默认 trace_config.csv),重建前与 manifest 持久
    字段 fail-closed 核对(_check_replay_config_against_manifest)。"""
    committed_ranks = {}     # (request_id, stage) -> set(rank)
    committed_counts = {}    # (request_id, stage) -> 节点总数
    watches = {}             # (request_id, stage) -> [watch, ...]
    kv_actions = []
    assignment_count = 0
    if not _seq_files(bridge_dir, "response_"):
        # (2026-10-01)裸 assert 在 python -O 下被剥离后会更晚以 TypeError
        # 崩溃——改显式 ValueError(fail-closed 通道不变)。
        if not manifest_path:
            raise ValueError(
                "bridge 无 response_*.json(消费即删);对账需要 --manifest "
                "做确定性 Python 侧重建")
        import sys as _sys
        _wl = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        for _p in (_wl, os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))):
            if _p not in _sys.path:
                _sys.path.insert(0, _p)
        from pathlib import Path as _Path
        from generate_face_trace import load_face_trace_config  # noqa: E402
        from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402
        from online.face_online_scheduler import FaceOnlineScheduler  # noqa
        with open(manifest_path, "r", encoding="utf-8") as _src:
            _manifest = json.load(_src)
        if config_path:
            _config = load_face_trace_config(_Path(config_path))
        else:
            _config = load_face_trace_config(
                _Path(os.path.join(_wl, "trace_config.csv")))
        _check_replay_config_against_manifest(_config, _manifest)
        _sched = FaceOnlineScheduler(
            manifest=_manifest, config=_config,
            graph=GraphBatchBuilder(_config), mode="strategy", sensing=True)
        for _seq, _delta in iter_request_records(bridge_dir):
            _batch = _sched.on_decision_batch(_delta)
            # 健康运行中 C++ 消费 response 即回 commit ack；重建侧必须同步
            # 回放合成 ack，否则 _emitted_by_delivery/_provisional_kv_actions
            # 只增不减，第 9 个交付即触发 propagating-tail fail-closed
            # （RuntimeError），对账在任何真实规模运行上必崩。ack 契约与
            # C++ 侧一致：schema v1、batch_id == delivery_sequence。
            _sched.on_commit_ack({
                "schema_version": 1,
                "batch_id": _delta["delivery_sequence"],
                "delivery_sequence": _delta["delivery_sequence"],
            })
            for node in _batch.get("nodes") or []:
                key = (node.get("request_id", ""), node.get("stage", ""))
                committed_ranks.setdefault(key, set()).add(int(node["rank"]))
                committed_counts[key] = committed_counts.get(key, 0) + 1
            for watch in _batch.get("watches") or []:
                key = (watch.get("request_id", ""), watch.get("stage", ""))
                watches.setdefault(key, []).append(watch)
            kv_actions.extend(_batch.get("kv_actions") or [])
            assignment_count += len(_batch.get("assignments") or [])
        return committed_ranks, committed_counts, watches, kv_actions, \
            assignment_count
    for seq in _seq_files(bridge_dir, "response_"):
        path = os.path.join(bridge_dir, "response_{}.json".format(seq))
        with open(path, "r", encoding="utf-8") as source:
            resp = json.load(source)
        for node in resp.get("nodes") or []:
            key = (node.get("request_id", ""), node.get("stage", ""))
            committed_ranks.setdefault(key, set()).add(int(node["rank"]))
            committed_counts[key] = committed_counts.get(key, 0) + 1
        for watch in resp.get("watches") or []:
            key = (watch.get("request_id", ""), watch.get("stage", ""))
            watches.setdefault(key, []).append(watch)
        kv_actions.extend(resp.get("kv_actions") or [])
        assignment_count += len(resp.get("assignments") or [])
    return committed_ranks, committed_counts, watches, kv_actions, \
        assignment_count


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


def load_cpp_phase_counters(cpp_log_path):
    """cpp.log 的 phase-4 end audit 与 phase-5 commit counters(可选;
    R2b/R2c 加固数据源,2026-08-16 wscllm 054118e 同款):
      phase-4 end audit: watch_registry_size/watch_stale/
        completed_facts_residual/affected_ranks_residual(全零 = watch
        生命周期干净收尾)
      phase-5 commit counters: total_nodes/total_watches(committed 批级
        权威计数,digest 双和对平目标)
    缺失行不在此报错(由对账项按可用性判定),返回 dict 缺省 None 字段。"""
    counters = {"phase4": None, "total_nodes": None, "total_watches": None}
    text = _read_cpp_log(cpp_log_path)
    if text is None:
        return counters
    match4 = re.search(
        r"\[online\] phase-4 end audit: watch_registry_size=(\d+) "
        r"watch_stale=(\d+) completed_facts_residual=(\d+) "
        r"affected_ranks_residual=(\d+)", text)
    if match4:
        counters["phase4"] = {
            "watch_registry_size": int(match4.group(1)),
            "watch_stale": int(match4.group(2)),
            "completed_facts_residual": int(match4.group(3)),
            "affected_ranks_residual": int(match4.group(4)),
        }
    match5 = re.search(
        r"\[online\] phase-5 commit counters:.*?\btotal_nodes=(\d+)\b"
        r".*?\btotal_watches=(\d+)\b", text)
    if match5:
        counters["total_nodes"] = int(match5.group(1))
        counters["total_watches"] = int(match5.group(2))
    return counters


def load_digests(run_bridge_dir):
    """graph_batch_digests.jsonl(批级 committed 摘要,R2b 加固数据源;
    2026-08-16 wscllm 054118e 同款)。可选数据源:生产默认 SH_GRAPH_
    DIGESTS=0 不产出本文件(C2/D2 2026-08-28 裁决,online_service 写出
    受该开关门控)。缺失/0 行返回 None,R2b 据此跳过(有 digest 才校验;
    2026-09-26 起,原"官方 runner 产出必含本文件"的 fail-closed 判据
    与生产默认矛盾,已废弃)。"""
    path = _artifact_path(run_bridge_dir, "graph_batch_digests.jsonl")
    if not os.path.exists(path):
        return None
    rows = list(_iter_jsonl(path))
    return rows or None


def _seq_files(bridge_dir, prefix):
    """列出 bridge 目录下 <prefix><seq>.json 的 seq(升序)。

    (2026-10-01)探测目录先经 _artifact_path 解析(live 优先、归档
    <run>/results/ 兜底)——与 ledger/sensing/digest 的口径一致,避免
    失败 run 的残留 response 已被归档到 results/ 时静默放弃真实
    committed 证据、转走确定性重建分支(重建会把包括未提交交付在内的
    全部 journal 行合成 ack,对账对象悄悄变成"重放的理想 run")。"""
    probe = _artifact_path(bridge_dir, "{}0.json".format(prefix))
    directory = os.path.dirname(os.path.abspath(probe))
    seqs = []
    for name in os.listdir(directory):
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
    parser.add_argument(
        "--config", default=None,
        help="trace_config.csv 路径(缺省用 workload 默认;确定性重建"
             "分支按它载入调度器配置,须与产生本 run 的配置一致,重建前"
             "按 manifest 持久字段 fail-closed 核对)")
    parser.add_argument("--cpp-log", default=None)
    parser.add_argument(
        "--expected-requests", type=int, default=1177,
        help="输入期望请求数(R0a;缺省 1177 = 20.csv 前 30s 验收值,"
             "跨输入复用时按物化实测改传,来源:face测试/开感知120报告"
             "_3mins.md 建议)")
    parser.add_argument(
        "--expected-accepted-sessions", type=int, default=112,
        help="输入期望 accepted 会话数(R0e;缺省 112 = 30s 验收值,同上)")
    parser.add_argument(
        "--max-rank", type=int, default=53,
        help="R2b 批级摘要 rank 上界(闭区间;缺省 53 = 54-NPU 验收 mesh,"
             "跨硬件复用时按 npus-1 改传)")
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
    committed_ranks, committed_counts, watches, kv_actions, \
        assignment_count = load_responses(
            args.bridge_dir, manifest_path=args.manifest,
            config_path=args.config)
    python_ledger = load_python_ledger(args.bridge_dir)
    query_log = load_sensing_query_log(args.bridge_dir)
    cpp_counters = load_cpp_counters(args.cpp_log)
    digests = load_digests(args.bridge_dir)
    phase_counters = load_cpp_phase_counters(args.cpp_log)

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

    # (2026-10-01 口径修正)流式改造后 ledger.jsonl 每行 layers 恒只含
    # "completed_unreconciled" 一个键(dump_ledger 在 ledger_sink 在场时
    # early-return,admitted/committed 中间层不再落盘)——按"有该键即计"
    # 的 dict 长度口径三值恒相等(数的是"有任何 fact 的请求数"),单项
    # 存在性假阴。改为只计对应 stage 真有 fact 的请求:三映射保持
    # rid->fact dict 形态(R1 逐请求生命周期对平要读 fact 载荷)但只收
    # 非 None 项,故 len() 即"有该 stage fact 的请求数",与 R0c 计数
    # 语义一致。
    prefill_facts = {
        rid: entry.get("prefill")
        for rid, entry in cpp_completions.items()
        if entry.get("prefill") is not None}
    decode_facts = {
        rid: entry.get("decode")
        for rid, entry in cpp_completions.items()
        if entry.get("decode") is not None}
    req_facts = {
        rid: entry.get("request_complete")
        for rid, entry in cpp_completions.items()
        if entry.get("request_complete") is not None}
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
        check("R0e", cpp_counters["accepted"] == args.expected_accepted_sessions,
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

    # ---- R2 committed 节点对平(watch 成员账本) ----
    # watch 成员 = 每 rank 最后一个该阶段节点(非全部阶段节点;阶段图可横跨
    # prefill+decode 两实例 rank——PD 分离并发 decode 结构,phase 2 B3 已验);
    # 因此成员数 != committed rank 数不是失配,正确核对是:
    #   R2a 每 (request, stage) 恰注册一次 watch,watch 成员数 == C++
    #       completed_groups node_count(request 文件 vs response 文件对平);
    #   R2b watch 成员 rank ⊆ 该 (request, stage) committed 节点 rank 集。
    member_bad = []
    watch_reg_bad = []
    for request_id in manifest_ids:
        for stage in ("prefill", "decode"):
            fact = cpp_completions.get(request_id, {}).get(stage)
            regs = watches.get((request_id, stage), [])
            if fact is None:
                continue
            if len(regs) != 1:
                watch_reg_bad.append({
                    "request_id": request_id, "stage": stage,
                    "watch_count": len(regs)})
                continue
            members = regs[0].get("members", {})
            if len(members) != fact.get("member_count", -1):
                member_bad.append({
                    "request_id": request_id, "stage": stage,
                    "member_count": fact.get("member_count"),
                    "watch_member_count": len(members),
                    "committed_ranks": sorted(committed_ranks.get(
                        (request_id, stage), set()))})
            member_ranks = set(int(rank) for rank in members)
            committed = committed_ranks.get((request_id, stage), set())
            if not member_ranks <= committed:
                member_bad.append({
                    "request_id": request_id, "stage": stage,
                    "members_outside_committed": sorted(
                        member_ranks - committed),
                    "committed_ranks": sorted(committed)})
    check("R2", not member_bad and not watch_reg_bad,
          "watch ledger mismatches: member_count/成员超集 {} 条,"
          "注册次数异常 {} 条".format(len(member_bad), len(watch_reg_bad)))

    # ---- R2b 批级摘要不变量(2026-08-16 加固,wscllm 054118e 同款;
    #      2026-09-29 M19 空性等价按列车化语义修正) ----
    # # (2026-09-30, ported from the face repo) 上界修正(py-online-E2):
    # watch_count ∈ [0, node_count] 且 ranks 为空 ⇒ watch_count==0(单向
    # 空性约束)。旧上界 watch_count <= len(ranks) 在列车化后结构性不
    # 成立:单实例批的 watch 按列车成员逐条注册(drain/exit 逐成员、
    # 哨兵、wakeup),同实例 ≥7 个 joiner 即合法出现 7 个 watch > 6
    # rank。结构正确的上界是 node_count——每个 watch 的标记节点真实
    # 落在本批(graph_batch_builder 的 _emit_train_tail_markers:每成员
    # 每 group rank 各 1 个标记节点),故 watch 数恒 ≤ 本批 committed
    # 节点数。空性保持单向(M19 口径不变):admission-only 批与纯逐出
    # mem 段合法地"有节点、零 watch",双向等价对健康 run 必误报;
    # ranks ⊆ [0, --max-rank](缺省 53 = 54-NPU 验收
    # mesh,跨硬件复用时改传);digest 双和与 C++ phase-5 权威计数对平
    # (sum(node_count)==total_nodes、sum(watch_count)==total_watches)。
    digest_bad = []
    if digests is None:
        # 2026-09-26 降级(R2 py-online-E1):生产默认 SH_GRAPH_DIGESTS=0
        # 不产出本文件(C2/D2 2026-08-28 裁决),原 fail-closed 判据使
        # 官方流程对账必然误报"不平"。改为"有 digest 才校验":数据源
        # 缺失时跳过本项(汇总行标 SKIP),不再计入 failures。
        pass
    else:
        node_sum = 0
        watch_sum = 0
        for row in digests:
            ranks = row.get("ranks") or []
            watch_count = row.get("watch_count", 0)
            node_count = row.get("node_count", 0)
            node_sum += node_count
            watch_sum += watch_count
            seq = row.get("delivery_sequence")
            if not 0 <= watch_count <= node_count:
                digest_bad.append({
                    "delivery_sequence": seq,
                    "violation": "watch_count 不在 [0, node_count]"
                                 "(列车化上界)",
                    "watch_count": watch_count, "node_count": node_count})
            if len(ranks) == 0 and watch_count != 0:
                digest_bad.append({
                    "delivery_sequence": seq,
                    "violation": "空批带 watch(ranks 空 ⇒ watch_count==0;"
                                 "watch_count==0 而 ranks 非空属列车化"
                                 " admission-only/纯逐出批,合法)",
                    "watch_count": watch_count, "ranks": ranks})
            if any(not 0 <= rank <= args.max_rank for rank in ranks):
                digest_bad.append({
                    "delivery_sequence": seq,
                    "violation": "rank 越界(ranks ⊄ [0,{}])".format(
                        args.max_rank),
                    "ranks": ranks})
        sum_bad = []
        if phase_counters["total_nodes"] is not None and \
                node_sum != phase_counters["total_nodes"]:
            sum_bad.append("sum(node_count)={} != cpp total_nodes={}".format(
                node_sum, phase_counters["total_nodes"]))
        if phase_counters["total_watches"] is not None and \
                watch_sum != phase_counters["total_watches"]:
            sum_bad.append("sum(watch_count)={} != cpp total_watches={}"
                           .format(watch_sum,
                                   phase_counters["total_watches"]))
        check("R2b", not digest_bad and not sum_bad,
              "批级摘要违例 {} 条{}".format(
                  len(digest_bad),
                  (";" + ";".join(sum_bad)) if sum_bad else
                  " (batches={}, node_sum={}, watch_sum={})".format(
                      len(digests), node_sum, watch_sum)))

    # ---- R2c phase-4 end audit 全零(watch 生命周期干净收尾) ----
    if phase_counters["phase4"] is not None:
        audit = phase_counters["phase4"]
        non_zero = {
            key: value for key, value in audit.items() if value != 0}
        check("R2c", not non_zero,
              "phase-4 end audit 非零: {} (期望全零: {})".format(
                  non_zero, audit))

    # ---- R3 KV 账本对平(Python KV 权威 vs C++ 完成事实) ----
    kv_by_request = {}
    for action in kv_actions:
        trigger = action.get("trigger_request_id")
        kv_by_request.setdefault(trigger, []).append(action)
    kv_bad = []
    kv_no_actions = [
        request_id for request_id in manifest_ids
        if not kv_by_request.get(request_id)]
    if kv_no_actions:
        kv_bad.append({"no_kv_actions": kv_no_actions[:10],
                       "count": len(kv_no_actions)})
    completion_event_bad = []
    kv_tick_bad = []
    for request_id in manifest_ids:
        completed_tick = (python_ledger.get(request_id, {})
                          .get("completed_unreconciled", {})
                          .get("completed_tick"))
        if completed_tick is None:
            continue
        events = kv_by_request.get(request_id, [])
        completion_events = [
            event for event in events
            if event.get("phase") == "completion"
            and event.get("event_type") == "retain_complete"]
        if not completion_events:
            completion_event_bad.append(request_id)
        else:
            # (2026-10-01)裸下标改 .get + 显式失配:坏行(缺
            # planner_time_ns)记入差异清单而非 KeyError traceback,
            # 取证质量与同文件其余检查一致。
            bad_ticks = [
                request_id for event in completion_events
                if not isinstance(event.get("planner_time_ns"), int)]
            if bad_ticks:
                completion_event_bad.append(request_id)
            else:
                max_tick = max(event["planner_time_ns"]
                               for event in completion_events)
                if max_tick != completed_tick:
                    kv_tick_bad.append({
                        "request_id": request_id,
                        "completion_kv_tick": max_tick,
                        "completed_tick": completed_tick})
        # (2026-10-01 补全)本行对全部事件取值,坏行(缺 planner_time_ns)
        # 也要落差异清单而非 KeyError;非整值按"晚于完成"无法判定处理,
        # 与 completion 侧 bad_ticks 同款并入 completion_event_bad。
        if any(not isinstance(event.get("planner_time_ns"), int)
               for event in events):
            completion_event_bad.append(request_id)
        elif any(event["planner_time_ns"] > completed_tick
                 for event in events):
            kv_tick_bad.append({
                "request_id": request_id,
                "event_after_completion": True,
                "completed_tick": completed_tick})
    check("R3", not kv_bad and not completion_event_bad and not kv_tick_bad,
          "KV ledger mismatches: no_actions={} missing_retain_complete={} "
          "tick_mismatch={}".format(
              len(kv_no_actions), len(completion_event_bad),
              len(kv_tick_bad)))

    # ---- R4 排队账本对平 ----
    # (2026-10-01 口径修正)流式改造后 ledger.jsonl 行的 layers 键集恒为
    # {"completed_unreconciled"}——admitted/committed 中间层不再落盘
    # (全仓唯一写方 _ledger_complete 只写 completed_unreconciled),
    # 原来的"无 admitted/committed 条目"检查结构性恒真(空转)。改为
    # 键集断言:任何额外层键(未来回归)立即失配;admitted/committed
    # 的运行末清空语义由 online_service 进程内 verify_run_end 承载
    # (失败即整 run 非零退出),本工具退化为形状守门。
    unexpected_layers = {
        rid: sorted(set(layers) - {"completed_unreconciled"})
        for rid, layers in python_ledger.items()
        if set(layers) - {"completed_unreconciled"}}
    check("R4a", not unexpected_layers,
          "ledger rows carry layers beyond completed_unreconciled: {!r}"
          .format(sorted(unexpected_layers.items())))
    # (2026-10-01 R2) R4b 独立化:原"admitted/committed 清空"检查在流式
    # 改造后结构性恒真,上一轮替换时误写成与 R4a 同一表达式(零判别力)。
    # 改为"每行都携带 completed_unreconciled 层"——缺层行是独立的形状
    # 违例(下游 R1 生命周期对平依赖该层存在)。
    missing_layer = sorted(
        rid for rid, layers in python_ledger.items()
        if "completed_unreconciled" not in layers)
    check("R4b", not missing_layer,
          "ledger rows missing the completed_unreconciled layer: {!r}"
          .format(missing_layer))
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
        # 末决策边界残差 = 最后请求 decode 的 end barrier 控制节点(列车
        # 迭代数字节的 all_reduce——拼 batch 改造后 barrier 携带 iterations
        # 字节,C++ CommCollective 的 coll.bytes 计入 per_request.comm_bytes,
        # 不再恒 1 字节;无 compute、无时长模型;在最终 watch 命中时仍在途,
        # 于运行结束前完成——C++ 干净退出 active=0 即证)。因此 R5a 断言:
        # 残差非空(终请求 barrier 必然在途)且全部为 barrier 签名——
        # 签名不钉 comm_bytes(末列车迭代数 ≥1 即合法,与 R7a2 同三项口径)。
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
                and group.get("estimated_remaining_ns") == 0)
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
        # (提交后逐批流入,完成时归零;整体上限 = 已提交总数)。
        over_committed = []
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
            if peak > committed:
                over_committed.append({
                    "request_id": request_id, "stage": stage,
                    "peak_injected": peak, "committed_nodes": committed})
        check("R5b", not over_committed,
              "{} (request, stage) injected peak > committed node "
              "count".format(len(over_committed)))
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
    # per_request 键集。拼 batch 列车化(2026-08-22)后的现实口径:列车
    # 共享体节点挂在批命名空间(train_id = "batch_train_*"),C++ 摘要的
    # per_request 组不再携带请求级键——请求级键集对平(py ⊆ cc)只在
    # 节点级请求归因存在的时代成立;本对账项随之同步为两层:
    #   R7a1 覆盖性:Python issued 非空 ⇔ C++ injected 未完成节点总数 > 0
    #     (Python 幽灵发射 = issued 非空而 C++ 全零,fail);
    #   R7a2 only-C++ 组合法性:barrier 签名(单节点/零 compute/零时长,
    #     watch 命中已报完成而控制尾节点在途的已知窗口)或批命名空间组
    #     (batch_train_ 前缀 = 列车共享体/哨兵,与调度器 BATCH_TRAIN_
    #     PREFIX 同源);非前缀且非 barrier 签名的请求级残差仍 fail;
    #   R7b 末边界 issued 清空(不变)。
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
        cc_total_nodes = 0
        for rank_entry in row.get("injected_unfinished", []):
            cc_total_nodes += rank_entry.get("node_count", 0)
            for group in rank_entry.get("per_request", []):
                cc_keys.add(group["request_id"])
                cc_groups.append(group)
        only_cpp = cc_keys - py_issued
        residual_peak = max(residual_peak, len(only_cpp))
        if py_issued and cc_total_nodes == 0:
            issued_only_python.append({
                "delivery_sequence": row["delivery_sequence"],
                "ghost_issued": sorted(py_issued)[:5]})
        # 残差核对:only_cpp 的每组须为 barrier 签名或批命名空间组。
        for group in cc_groups:
            if group["request_id"] not in only_cpp:
                continue
            signature_ok = (
                (group.get("node_count", 0) == 1
                 and group.get("compute_ops", 0) == 0
                 and group.get("estimated_remaining_ns", 0) == 0)
                or group["request_id"].startswith(BATCH_TRAIN_PREFIX))
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
          "{} boundaries with Python-issued but zero C++ unfinished nodes "
          "(ghost issue; per-request key parity ended with the batched-"
          "train namespace)".format(len(issued_only_python)))
    check("R7a2", not issued_residual_bad,
          "{} residual groups neither barrier-signed nor batch-train-"
          "namespaced in the only-C++ window".format(len(issued_residual_bad)))
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
    if args.report:
        with open(args.report, "w", encoding="utf-8") as out:
            out.write(_render_report(
                balanced=balanced, failures=failures, manifest_count=len(
                    manifest_ids),
                expected_requests=args.expected_requests,
                expected_sessions=args.expected_accepted_sessions,
                arrival_epochs=arrival_epochs,
                delivery_epochs=len(query_log),
                assignment_count=assignment_count,
                kv_count=len(kv_actions),
                cpp_counters=cpp_counters,
                lifecycle_bad=lifecycle_bad, member_bad=member_bad,
                watch_reg_bad=watch_reg_bad,
                digest_bad=(digest_bad if digests is not None else None),
                phase4_audit=(phase_counters["phase4"]
                              if phase_counters["phase4"] is not None
                              else None),
                kv_bad=kv_bad, kv_no_actions=kv_no_actions,
                completion_event_bad=completion_event_bad,
                kv_tick_bad=kv_tick_bad,
                unexpected_layers=unexpected_layers,
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


def _render_report(balanced, failures, manifest_count, expected_requests,
                   expected_sessions, arrival_epochs,
                   delivery_epochs, assignment_count, kv_count, cpp_counters,
                   lifecycle_bad, member_bad, watch_reg_bad, digest_bad,
                   phase4_audit, kv_bad,
                   kv_no_actions, completion_event_bad, kv_tick_bad,
                   unexpected_layers, ready_bad, final_ready,
                   issued_only_python, issued_residual_bad, residual_peak,
                   final_issued, cc_internal_bad, cc_cover_bad,
                   peak_node, peak_in_flight, peak_free):
    lines = []
    lines.append("# 结束总账核对报告 — 分层账本逐层对平(阶段 7 §10.1,2026-08-15)")
    lines.append("")
    lines.append("> 方案 §6.2 操作 3 / contract ⑥ + §10.1:Python KV/排队账本 vs C++")
    lines.append("> 执行事实,逐 request/stage 对平;八层 remaining-load 账本")
    lines.append("> 逐层对账(R0-R8)。输入:strategy 感知模式运行({} requests;"
                 "输入期望".format(manifest_count))
    lines.append("> --expected-requests={} / --expected-accepted-sessions={}"
                 .format(expected_requests, expected_sessions))
    lines.append("> ;30s 验收缺省值 1177/112,跨输入复用时按物化实测改传)。")
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
        "PASS" if not any(item.startswith("R0") for item, _ in failures)
        else "FAIL"))
    lines.append("| R1 逐 request 生命周期对平(admitted<=commit<=prefill<=decode==complete;gen 0/1/1) | {} |".format(
        "PASS" if not lifecycle_bad else "FAIL: {} 条".format(len(lifecycle_bad))))
    lines.append("| R2 committed 节点对平(每 (request, stage) 恰注册一次 watch;watch 成员数 == completed_groups node_count;成员 rank ⊆ committed rank 集) | {} |".format(
        "PASS" if not (member_bad or watch_reg_bad)
        else "FAIL: 成员 {} 条 / 注册 {} 条".format(
            len(member_bad), len(watch_reg_bad))))
    lines.append("| R2b 批级摘要不变量(watch_count ∈ [0,node_count] 列车化上界;ranks 空 ⇒ watch_count==0 单向空性;ranks ⊆ [0,--max-rank](缺省 53);digest 双和 == cpp phase-5 total_nodes/total_watches;2026-08-16 加固,wscllm 054118e 同款;有 digest 才校验) | {} |".format(
        # digest_bad=None = 数据源缺失(main 传入),即 SKIP;此处不得引用
        # main 的局部 digests(NameError,2026-09-26 R2b 降级批引入)。
        "SKIP(graph_batch_digests.jsonl 缺失;生产默认 SH_GRAPH_DIGESTS=0 "
        "不产出)" if digest_bad is None else
        "PASS" if not digest_bad and not any(
            "R2b" in item for item, _ in failures)
        else "FAIL: {} 条违例/失配".format(len(digest_bad))))
    lines.append("| R2c phase-4 end audit 全零(watch 生命周期干净收尾) | {} |".format(
        "PASS" if phase4_audit is not None and not any(
            value for value in phase4_audit.values())
        and not any("R2c" in item for item, _ in failures)
        else "FAIL" if phase4_audit is not None else "SKIP(cpp.log 无 phase-4 行)"))
    lines.append("| R3 KV 账本对平(每 request 有 kv 动作;completion retain_complete 事件 tick == completed_tick;无事件晚于完成) | {} |".format(
        "PASS" if not (kv_bad or completion_event_bad or kv_tick_bad)
        else "FAIL: no_actions={} retain_complete={} tick={}".format(
            len(kv_no_actions), len(completion_event_bad),
            len(kv_tick_bad))))
    lines.append("| R4 排队账本对平(流式 ledger 行键集恒 completed_unreconciled;admitted/committed 清空由 verify_run_end 承载;末边界 admitted_count=0) | {} |".format(
        "PASS" if not unexpected_layers
        and not any(item.startswith("R4") for item, _ in failures)
        else "FAIL: unexpected_layers={} ({})".format(
            len(unexpected_layers),
            sorted(unexpected_layers)[:5])))
    lines.append("| R5 injected-unfinished 对平(末决策边界残差 = 终请求 decode end-barrier 控制节点;注入峰值 ≤ committed 节点数) | {} |".format(
        "PASS" if not any(item.startswith("R5") for item, _ in failures)
        else "FAIL"))
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
    lines.append("- manifest requests: {};arrival epochs: {};delivery epochs: {};assignments: {};kv_actions 总条数: {};cpp.log counters: {}".format(
        manifest_count, arrival_epochs, delivery_epochs, assignment_count,
        kv_count, cpp_counters if cpp_counters else "n/a"))
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
    lines.append("| remote-memory FIFO | 不适用(池端口为并发流体计时,无 FIFO 等待账本;远端内存后端已启用,README §D) | 显式占位,无账本 |")
    lines.append("| network pending/active | C++ 执行事实(injected-unfinished 摘要) | 对平 R8(计数/审计) |")
    lines.append("| local HBM job | 不适用(对账八层无 HBM 作业账本;LocalHbmBandwidthModel 已随底座启用,README §G) | 显式占位,无账本 |")
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
                 "<run>/cpp.log --config <trace_config.csv(与 run 同配置,"
                 "缺省 workload 默认)> --expected-requests N "
                 "--expected-accepted-sessions N --report <out>.md`;"
                 "退出 0 = 对平;期望常量缺省 1177/112(30s 验收值)。")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
