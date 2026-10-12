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
                              train_ledger.jsonl(列车发射账本:逐列车
                              joiners/exits/drains 与发射 tick——R7a1/R8b
                              列车成员时间线回放的数据源)
                              cpp.log 的 phase-5 commit counters 与
                              phase-4 end audit(committed/watch/kv 权威计数)

数据源入口自检(fail-closed,2026-09-27):生产缺省两数据源皆不产出——
SH_GRAPH_DIGESTS 缺省 0(graph_batch_digests.jsonl 不写)、
ASTRA_SIM_ONLINE_COMPLETED_NODES 缺省非 exact(request 文件
completed_nodes 恒空)。缺失/为空即退出 2,报出所需运行环境变量;禁止在
残缺数据源上产出误导性"不平"报告或裸 FileNotFoundError 崩溃。

batch_train_ 口径(2026-09-27):request_id 带 batch_train_ 前缀的完成
事实 = 列车哨兵/首步唤醒 watch 的批命名空间事件(调度器消费侧按前缀
剥离,见 online_scheduler_base._settle_completions / 变体
run_variant_policy),对账侧同样剥离——不计入请求级 prefill/decode
完成事实与 committed 对,单独计数供 R2a watch 总数对平(每个
batch_train_ watch 恰 fire 一次 = 恰一个完成组)。
  --manifest <ET_DIR>/manifest.json   全部请求事实(1177 requests)
  --cpp-log <run>/cpp.log     C++ 门计数器与 phase-5/4 审计计数器(必填;
                              缺省即拒绝——缺它 R0d/R0e/R2c/R3a/R3c 的
                              `is not None` 门全部跳过 = 空洞通过,违反
                              权威数据源契约;归档态透明回退 cpp.log.gz)
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
     恰注册一次 watch —— C++ total_watches == 真实请求 completed_nodes
     对计数 + batch_train_ 哨兵/唤醒 watch 对(batch_train_ 批命名空间
     watch 与请求级 watch 同计数器,双口径合并对平)== ledger stage 对
     计数 + 哨兵/唤醒对 == sum(digests.watch_count);批级摘要合法(watch
     每批 watch_count ∈ [0, node_count] 且与 ranks 空性一致、ranks ⊆ [0,53]、sum(node_count) == C++ total_nodes);
     watch 生命周期干净收尾(phase-4 end audit 全零)。
  R3 KV 账本对平(phase-7 契约):每 request 恰一条 completion 决策
     (online_decision_log, tick = KV 生命周期终止时刻)且 C++
     total_kv_actions >= 请求数(每 request ≥ 1 动作的计数级保证);
     completion tick == ledger completed_tick;prefill<=decode<=completion
     顺序;kv_instance_after_completion ∈ [0,53] 或 None(B2 三态:REMOTE
     态会话无本地实例,None = 纯远端/终局核销后的合法取值);运行结束
     active=0。
  R4 排队账本对平:运行结束 admitted/committed 层已清空(ledger.jsonl 无
     admitted/committed 条目);最后一决策边界 admitted_count == 0。
  R5 injected-unfinished 对平(sensing_query_log):末决策边界残差 = 最后
     完成动作的 end-barrier 控制尾节点(无 compute、无时长模型;最终
     watch 命中时仍在途,运行结束前完成——C++ 干净退出 active=0 即证)。
     列车化口径(2026-10-01):decode 的 end barrier 是列车共享节点
     (batch_train_ 批命名空间,不再按命名空间裁剪),统一按 barrier 签名
     断言(decode 阶段、单节点、零 compute、零剩余时长、控制字量级
     comm);断言残差非空且全部为 barrier 签名;注入峰值(节点数,跨
     rank/代求和)≤ committed 节点数(以 completed_nodes 逐对计数为
     上限;提交后逐批流入,完成时归零)。
  R6 ready 层对平(§10.1):每边界 ready(就绪可服务排队成员,策略 override
     视图)⊆ admitted(键集)且 ready ∩ issued == ∅;末决策边界 ready==0。
  R7 issued 层对平(§10.1,核心;列车感知口径 2026-10-01):每边界 Python
     issued 请求(emit 登记、completion 核销于策略完成处理前)须有 C++
     物理在飞证据,二者居一——请求级未完成组(joiner 迁移/exit 标记;
     prefill 段与 exit 标记在途窗口),或所在列车的批命名空间未完成组
     (列车体运行期间;成员关系由 train_ledger.jsonl 回放)。反向 only-C++
     残差全部为 barrier 控制尾签名(请求级逐组核对;批命名空间组恒落
     only_cpp 窗口、不参与该方向核对,与 R5 同源的已知在途窗口);末
     决策边界 issued==0。
  R8 network pending/active 计数对账(§10.1):C++ 摘要内部一致(每 rank
     node_count >= in_flight_node_count + free_node_count,剩余 = 依赖未
     满足的未发射中间节点);列车可见性(列车化口径):每个当前列车
     (发射于本边界 tick 前、同实例后继尚未发射)只要有仍在 issued 的
     成员,就必须在 C++ 摘要出现(体/哨兵/共享 barrier 至少一节点未
     完成);峰值统计列示。

八层账本口径(§10.1):admitted / committed / ready / issued / network
pending/active 参与对平;remote 层与 local HBM 为显式占位、列示不参与
逐层对平(py-metrics-C2 修正 2026-09-27:措辞按本仓实际机制改写,豁免
依据不再是基线遗留的"wscllm 无远端内存后端 FIFO"——本仓远端共享 KV
池在案,边缘 NPU 流体端口承载 remote_store 整体逐出 / remote_load
全量回迁,逐事务事实落 bridge_dir/remote_memory_transactions.jsonl;
缺的是逐请求 FIFO job 账本,远端占用以远端账面单列、以该逐事务明细为
权威事实。local HBM 计时模型已激活——comm 边端点计费 hbm_charge 在案
(裁决 #35),缺的是独立 local-HBM 节点账本)。

退出码:0 = 对平;1 = 存在失配(报告逐项列出差异与证据);2 = 数据源
缺失/为空(fail-closed,不产出报告)。"""

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
from online_scheduler_base import (  # noqa: E402
    BATCH_TRAIN_PREFIX,
    STAGE_DECODE,
    STAGE_PREFILL,
)


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


def _resolve_cpp_log_path(cpp_log_path):
    """cpp.log 路径解析(含归档 .gz 透明回退);缺失/不存在返回 None。"""
    if not cpp_log_path:
        return None
    path = cpp_log_path
    if not os.path.isfile(path) and not path.endswith(".gz") \
            and os.path.isfile(path + ".gz"):
        path += ".gz"
    if not os.path.isfile(path):
        return None
    return path


def _read_cpp_log(cpp_log_path):
    """Read cpp.log or its transparent post-archive cpp.log.gz form."""
    path = _resolve_cpp_log_path(cpp_log_path)
    if path is None:
        return None
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as source:
        return source.read()


def _iter_jsonl(path):
    # (2026-10-01, E4 复审) 坏行(截断/归档损坏/代际不符)原先以裸
    # json.JSONDecodeError traceback + 退出码 1 收场,违反本工具 0/1/2
    # 退出码契约(1 = 已产出差异报告的失配);与 slo_common.iter_jsonl 的
    # fail-closed=2 范式对齐,此处按数据源损坏 exit 2。
    with open(path, "r", encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    print("[ledger_reconcile] FAIL-CLOSED: jsonl 坏行: "
                          "{}:{} ({}: {})".format(path, line_no,
                                                  type(exc).__name__, exc),
                          file=sys.stderr)
                    sys.exit(2)


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
      duplicate_pairs: 同一 (request, stage) 重复完成的键(失配证据,
                       R0c6 消费;此前为裸 AssertionError,不进报告)。
    batch_train_ 前缀(列车哨兵/首步唤醒 watch 的批命名空间事件)不入
    请求级完成事实,其完成组由 load_committed 单独收集供 R2a watch 总数
    对平。完成事实去重键 = (request, stage)(不含 generation;R1 的生成代
    期望由 R1 单独断言)。"""
    completions = {}
    arrival_epochs = 0
    duplicate_pairs = []
    for seq, req in iter_request_records(bridge_dir):
        arrival_epochs += len(req.get("arrivals") or [])
        for group in req.get("completed_groups") or []:
            request_id = group["request_id"]
            stage = group["stage"]
            if request_id.startswith(BATCH_TRAIN_PREFIX):
                continue  # 批命名空间控制事件:不入 manifest 请求域
            fact = {
                "tick": req["tick"],
                "generation": group.get("generation", 0),
                "member_count": group.get("node_count", 0),
                "delivery_sequence": seq,
            }
            entry = completions.setdefault(request_id, {})
            key = stage if stage else "request_complete"
            if key in entry:
                duplicate_pairs.append({
                    "request_id": request_id,
                    "stage": stage,
                    "delivery_sequences": [
                        entry[key]["delivery_sequence"], seq]})
                continue
            entry[key] = fact
    return completions, arrival_epochs, duplicate_pairs


def load_committed(bridge_dir):
    """phase-7 契约(request 文件 = C++ 完成事实 + 逐节点完成明细;
    response 文件消费即删,committed 侧数据源迁移至 request 文件):
      committed_counts: (request_id, stage) -> completed_nodes 总数
                        (仅真实请求;batch_train_ 前缀剥离,M8-B2)
      completed_nodes_total: 真实请求的 completed_nodes 条数(批命名空间
                        哨兵/唤醒节点同口径剥离)
      batch_train_watch_pairs: 哨兵/唤醒 watch 完成组 (request_id, stage)
                        对集合(每 watch 恰 fire 一次,供 R2a watch
                        总数对平)
    completed_nodes 即旧 response 文件 "nodes"(逐节点 rank/node_id),
    terminal_status=0 表示正常完成。"""
    committed_counts = {}
    completed_nodes_total = 0
    batch_train_watch_pairs = set()
    for seq, req in iter_request_records(bridge_dir):
        for group in req.get("completed_groups") or []:
            request_id = group["request_id"]
            if request_id.startswith(BATCH_TRAIN_PREFIX):
                if group["stage"] in (STAGE_PREFILL, STAGE_DECODE):
                    batch_train_watch_pairs.add(
                        (request_id, group["stage"]))
                continue
        for node in req.get("completed_nodes") or []:
            request_id = node.get("request_id", "")
            if request_id.startswith(BATCH_TRAIN_PREFIX):
                continue
            key = (request_id, node.get("stage", ""))
            committed_counts[key] = committed_counts.get(key, 0) + 1
            completed_nodes_total += 1
    return committed_counts, completed_nodes_total, batch_train_watch_pairs


def load_digests(bridge_dir):
    """graph_batch_digests.jsonl(批级 committed 摘要):每批 watch_count
    ∈ [0, node_count] 且 watch_count==0 ⇔ ranks 空,ranks = 批内 committed
    rank 集,node_count = 批节点总数。
    sum(node_count) == C++ total_nodes;sum(watch_count) == C++ total_watches
    (两侧均含 batch_train_ 批命名空间 watch,每 (request, stage) 恰注册
    一次 watch 的批级等价)。文件缺失/为空由 main 入口自检 fail-closed
    (exit 2),本函数不做存在性分支。"""
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
    三条决策(kind);completion 行 tick 即 KV 生命周期终止时刻。
    同一 (kind, request_id) 重复出现记入 duplicates(M32 修复:此前 dict
    赋值 last-wins 静默覆盖重复行,R3 只查缺失不查重复/总数恒 PASS)。
    文件缺失/为空由 main 入口自检 fail-closed(exit 2),本函数不做
    存在性分支。"""
    by_kind = {"prefill": {}, "decode": {}, "completion": {}}
    kind_counts = {}
    duplicates = []
    rows = 0
    for row in _iter_jsonl(_artifact_path(
            bridge_dir, "online_decision_log.jsonl")):
        rows += 1
        kind = row.get("kind")
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        if kind in by_kind:
            request_id = row["request_id"]
            if request_id in by_kind[kind]:
                duplicates.append({
                    "kind": kind,
                    "request_id": request_id,
                    "ticks": [by_kind[kind][request_id].get("tick"),
                              row.get("tick")]})
            by_kind[kind][request_id] = row
    return by_kind, kind_counts, rows, duplicates


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
    completed_unreconciled}(各层可能缺省)。
    每 request 恰一条 ledger 行(G7-10 修复:dict 赋值 last-wins 此前会
    静默吞掉重复完成行;现去重前记数比对,总行数与重复 request 由
    R0b3 对平)。文件缺失/为空由 main 入口自检 fail-closed(exit 2),
    本函数不做存在性分支。"""
    ledger = {}
    duplicates = []
    rows = 0
    for row in _iter_jsonl(_artifact_path(bridge_dir, "ledger.jsonl")):
        rows += 1
        request_id = row["request_id"]
        if request_id in ledger:
            duplicates.append(request_id)
        ledger[request_id] = row.get("layers", {})
    return ledger, rows, duplicates


def load_sensing_query_log(bridge_dir):
    """sensing_query_log.jsonl 行列表。文件缺失由 main 入口自检
    exit 2(与 digests/decision_log/ledger.jsonl 同款 fail-closed);
    文件在场但为空仍返回空列表——R5a 的 "empty" 分支是设计内的红路径。
    不裸 FileNotFoundError。"""
    path = _artifact_path(bridge_dir, "sensing_query_log.jsonl")
    if not os.path.isfile(path):
        return []
    return list(_iter_jsonl(path))


def load_train_ledger(bridge_dir):
    """train_ledger.jsonl 行列表(列车发射账本)。文件缺失由 main 入口
    自检 exit 2;在场但为空(零 decode 工作的运行)返回空列表——列车
    覆盖退化为纯请求级键核对,语义自洽。不裸 FileNotFoundError。"""
    path = _artifact_path(bridge_dir, "train_ledger.jsonl")
    if not os.path.isfile(path):
        return None
    return list(_iter_jsonl(path))


def build_train_membership(train_ledger_rows):
    """逐 decode 实例回放列车成员时间线(仅 batch_train_ 行;文件序即
    发射序,实例内发射 tick 单调——实例忙门保证一列车在飞)。

    每列车输出 {"train_id", "tick", "instance", "members",
    "successor_tick"}:members = 该列车发射时刻的实例成员快照(含本车
    joiners;本车 exits 成员的 exit 标记节点属于本列车,故先记快照、
    快照之后才移除 exits/drains);successor_tick = 同实例下一列车发射
    tick(末列车 None;逆序一次扫描解析,线性复杂度)。

    prefill_train 行不参与:prefill 段物理节点是请求级命名,issued 覆盖
    由请求级键直接给出,无需列车回放。"""
    trains = []
    active_by_instance = {}
    for row in train_ledger_rows:
        train_id = row.get("train_id", "")
        if not train_id.startswith(BATCH_TRAIN_PREFIX):
            continue
        instance = row.get("instance_index")
        active = active_by_instance.setdefault(instance, set())
        snapshot = active | set(row.get("joiners") or ())
        trains.append({
            "train_id": train_id,
            "tick": row.get("tick", 0),
            "instance": instance,
            "members": snapshot,
            "successor_tick": None,
        })
        active_by_instance[instance] = (
            snapshot
            - set(row.get("exits") or ())
            - set(row.get("drains") or ()))
    successor_by_instance = {}
    for train in reversed(trains):
        train["successor_tick"] = successor_by_instance.get(
            train["instance"])
        successor_by_instance[train["instance"]] = train["tick"]
    return trains


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


# ---------------------------------------------------------------------------
# 对账
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="phase-3 ledger reconciliation (strategy sensing run)")
    parser.add_argument("--bridge-dir", required=True)
    parser.add_argument("--manifest", required=True)
    # P2-19(2026-09-28):--cpp-log 必填——R0d/R0e(service counters)与
    # R2c/R3a/R3c(phase-5/4 audit)全以 cpp.log 审计计数器为权威数据源,
    # 缺省 None 时全部 `is not None` 门跳过 = 空洞通过(报告仍打 PASS),
    # 违反头注权威数据源契约。缺参由 argparse 报错退出非 0;路径给出但
    # 文件不存在(含 .gz 回退后)同样 fail-closed(入口自检退出 2),防
    # 拼写错路径退化回空洞。
    parser.add_argument(
        "--cpp-log", required=True,
        help="C++ 门/审计计数器权威数据源(<run>/cpp.log,归档态给 "
             "cpp.log.gz);R0d/R0e/R2c/R3a/R3c 的对平依据(必填)")
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

    # ---- manifest 存在性自检(fail-closed)----
    if not os.path.isfile(args.manifest):
        print("[ledger_reconcile] FAIL-CLOSED: manifest 缺失: {} "
              "(用法:--manifest <ET_DIR>/manifest.json)".format(
                  args.manifest), file=sys.stderr)
        return 2

    # ---- P2-19(2026-09-28):cpp.log 权威数据源存在性自检 ----
    # --cpp-log 已必填;路径给出但文件不存在(含 .gz 回退后)若静默退回
    # None,R0d/R0e/R2c/R3a/R3c 的 `is not None` 门全部跳过 = 空洞通过——
    # 与数据源缺失同款 fail-closed(exit 2,不产出报告)。
    if _resolve_cpp_log_path(args.cpp_log) is None:
        print("[ledger_reconcile] FAIL-CLOSED: cpp.log 不存在: {} "
              "—— R0d/R0e(service counters)与 R2c/R3a/R3c(phase-5/4 "
              "audit)以 cpp.log 审计计数器为权威数据源,缺失即全部空洞"
              "通过;传 --cpp-log <run>/cpp.log(归档态给 cpp.log.gz)"
              .format(args.cpp_log), file=sys.stderr)
        return 2

    # ---- P2-19 补口(2026-10-01):文件在场但计数器行缺失 ----
    # 截断/归档损坏/C++ 侧输出措辞漂移/未跑到 phase-4/5 的日志会让
    # load_cpp_counters / load_cpp_audit 返回 None 或缺组,原先
    # R0d/R0e/R2c/R3a/R3c 的 `is not None` 门随之全部跳过 = 空洞通过
    # (vacuous pass,报告仍打 PASS)。与"文件缺失"同款 fail-closed(exit 2,
    # 不产出报告);放在其余数据源入口自检之前——本门只依赖 cpp.log 自身。
    # cpp_audit 须同时含 m5(phase-5 commit counters)与 m4(phase-4 end
    # audit)两组键:任一组正则未命中即视为审计行缺失。
    cpp_counters = load_cpp_counters(args.cpp_log)
    cpp_audit = load_cpp_audit(args.cpp_log)
    if (cpp_counters is None or cpp_audit is None
            or "graph_batch_count" not in cpp_audit
            or "watch_registry_size" not in cpp_audit):
        print("[ledger_reconcile] FAIL-CLOSED: cpp.log 在场但缺少 "
              "[online] service counters / phase-5 commit counters / "
              "phase-4 end audit 行(截断、归档损坏或日志代际不符): {}"
              .format(args.cpp_log), file=sys.stderr)
        return 2

    # ---- 数据源入口自检(M8-B1 修复 2026-09-27:fail-closed)----
    # graph_batch_digests.jsonl 生产缺省不写(SH_GRAPH_DIGESTS 缺省 0),
    # 缺失时 load_digests 只会裸 FileNotFoundError;缺失/为空即在入口
    # 报出所需运行环境变量并退出 2(fail-closed 契约),禁止在残缺数据源
    # 上继续产出误导性结果。
    digests_path = _artifact_path(args.bridge_dir,
                                  "graph_batch_digests.jsonl")
    if not os.path.isfile(digests_path) or \
            os.path.getsize(digests_path) == 0:
        print("[ledger_reconcile] FAIL-CLOSED: graph_batch_digests.jsonl "
              "缺失/为空: {} —— R2a/R2b 的 watch 总数与批级摘要无对平数据"
              "源(该产物生产缺省不写,运行需 SH_GRAPH_DIGESTS=1)"
              .format(digests_path), file=sys.stderr)
        return 2

    # M8-B2 补口(2026-09-27):decision_log 与 ledger.jsonl 的加载路径
    # (_iter_jsonl)没有存在性检查,缺失时会在 load 阶段裸 FileNotFoundError
    # 崩溃,违反头注"禁止裸 FileNotFoundError"与退出码契约(2 = 数据源
    # 缺失/为空)。与 digests 同款入口自检。
    decision_log_path = _artifact_path(args.bridge_dir,
                                       "online_decision_log.jsonl")
    if not os.path.isfile(decision_log_path) or \
            os.path.getsize(decision_log_path) == 0:
        print("[ledger_reconcile] FAIL-CLOSED: online_decision_log.jsonl "
              "缺失/为空: {} —— R3 的 KV 生命周期对平无决策数据源"
              .format(decision_log_path), file=sys.stderr)
        return 2
    # (2026-10-01, D1 复审) 非空但零可解析记录(纯空白文件)同样拒绝:
    # size>0 门放行后 decision_rows==0,R3 块整体跳过而 _render_report 的
    # kv_low/active_left 形参悬空——NameError 或空洞通过两态皆违契约。
    if sum(1 for _ in _iter_jsonl(decision_log_path)) == 0:
        print("[ledger_reconcile] FAIL-CLOSED: online_decision_log.jsonl "
              "在场但零条可解析记录: {}".format(decision_log_path),
              file=sys.stderr)
        return 2
    # sensing_query_log.jsonl 缺失同款入口自检(在场但为空不拦——R5a 的
    # empty 分支是设计内红路径,会以 FAIL 判定呈现)。
    query_log_path = _artifact_path(args.bridge_dir,
                                    "sensing_query_log.jsonl")
    if not os.path.isfile(query_log_path):
        print("[ledger_reconcile] FAIL-CLOSED: sensing_query_log.jsonl "
              "缺失: {} —— R5a/R6/R7/R8 的 ready/issued 层对平无感知"
              "查询数据源(该产物仅感知运行产出)".format(query_log_path),
              file=sys.stderr)
        return 2
    # train_ledger.jsonl 同款入口自检(列车感知口径,2026-10-01):R7a1 的
    # 列车覆盖与 R8b 的列车可见性以列车成员时间线为数据源;strategy 运行
    # 恒产出该账本(runner 归档至 results/)。在场但为空不拦——零 decode
    # 工作的运行合法,覆盖退化为纯请求级核对。
    train_ledger_path = _artifact_path(args.bridge_dir, "train_ledger.jsonl")
    if not os.path.isfile(train_ledger_path):
        print("[ledger_reconcile] FAIL-CLOSED: train_ledger.jsonl "
              "缺失: {} —— R7a1/R8b 的列车成员时间线无数据源"
              "(strategy 运行恒产出该账本)".format(train_ledger_path),
              file=sys.stderr)
        return 2
    ledger_path = _artifact_path(args.bridge_dir, "ledger.jsonl")
    if not os.path.isfile(ledger_path) or \
            os.path.getsize(ledger_path) == 0:
        print("[ledger_reconcile] FAIL-CLOSED: ledger.jsonl "
              "缺失/为空: {} —— R0b/R1/R4 的 Python 账本对平无账本数据源"
              "(该产物仅感知运行产出)"
              .format(ledger_path), file=sys.stderr)
        return 2

    failures = []  # (item, evidence)
    # Report-only observations (C++ "reported, not asserted" counters):
    # carried into the report evidence, never into the verdict.
    report_facts = []

    def check(item, cond, evidence):
        if not cond:
            failures.append((item, evidence))
            return False
        return True

    # ---- R0 集合对平 ----
    manifest_ids = load_manifest(args.manifest)
    cpp_completions, arrival_epochs, cpp_duplicate_pairs = \
        load_cpp_facts(args.bridge_dir)
    committed_counts, completed_nodes_total, batch_train_watch_pairs = \
        load_committed(args.bridge_dir)
    if cpp_completions and completed_nodes_total == 0:
        # 只补 digests 的组合:completed_nodes 恒空(ASTRA_SIM_ONLINE_
        # COMPLETED_NODES 缺省非 exact),committed_pairs 为空会令
        # R2a/R5b 必然误报不平——入口即拒绝而非产出垃圾报告。
        print("[ledger_reconcile] FAIL-CLOSED: request 文件 completed_nodes "
              "恒空而 C++ 完成事实存在 —— 逐对 committed 上限无数据源"
              "(运行需 ASTRA_SIM_ONLINE_COMPLETED_NODES=exact)",
              file=sys.stderr)
        return 2
    digests, watch_total, digest_node_total = load_digests(args.bridge_dir)
    decision_log, decision_kind_counts, decision_rows, decision_duplicates = \
        load_decision_log(args.bridge_dir)
    python_ledger, ledger_rows, ledger_duplicates = load_python_ledger(
        args.bridge_dir)
    query_log = load_sensing_query_log(args.bridge_dir)
    # 列车成员时间线(R7a1 列车覆盖 / R8b 列车可见性的数据源;入口自检
    # 已保证账本在场,None 分支不可达,防御式回退空列表)。
    trains_by_emission = build_train_membership(
        load_train_ledger(args.bridge_dir) or [])
    trains_of_request = {}
    for train in trains_by_emission:
        for member in train["members"]:
            trains_of_request.setdefault(member, []).append(train)
    # cpp_counters/cpp_audit 由入口门解析并复用(2026-10-01 D1 复审:
    # 原先在此二次读取,把 cpp.log 全文/gz 重复解析了一遍)。

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
    check("R0b3", ledger_rows == len(python_ledger)
          and not ledger_duplicates,
          "ledger.jsonl 行数 {} != 唯一 request 数 {}(重复完成行会被 dict "
          "赋值 last-wins 静默覆盖,G7-10 修复后对平): {!r}".format(
              ledger_rows, len(python_ledger),
              sorted(set(ledger_duplicates))[:5]))

    prefill_facts = {
        rid: entry.get("prefill") for rid, entry in cpp_completions.items()}
    decode_facts = {
        rid: entry.get("decode") for rid, entry in cpp_completions.items()}
    req_facts = {
        rid: entry.get("request_complete")
        for rid, entry in cpp_completions.items()}
    # M30 修复:R0c/R0c1/R0c2 此前是同一断言(字典推导保留全部键,三个
    # len 恒等于 len(cpp_completions),任一 stage 缺失都检不出)。改为真正
    # 分阶段度量:按 stage 分组计数后分别断言(batch_train_ 批命名空间
    # 已在 load_cpp_facts 剔除,计数不含哨兵)。归档格式自带 stage 维度
    # (completed_groups[].stage,stage=='' 映射 request_complete)。
    stage_fact_counts = {"prefill": 0, "decode": 0, "request_complete": 0}
    for entry in cpp_completions.values():
        for stage in entry:
            if stage in stage_fact_counts:
                stage_fact_counts[stage] += 1
    check("R0c", stage_fact_counts["request_complete"] == manifest_count,
          "C++ REQUEST_COMPLETE facts = {} (expected {};按 stage 分组计数,"
          "batch_train_ 批命名空间已剔除)".format(
              stage_fact_counts["request_complete"], manifest_count))
    check("R0c1", stage_fact_counts["prefill"] == manifest_count,
          "C++ prefill completion facts = {} (expected {};按 stage 分组计数,"
          "batch_train_ 批命名空间已剔除)".format(
              stage_fact_counts["prefill"], manifest_count))
    check("R0c2", stage_fact_counts["decode"] == manifest_count,
          "C++ decode completion facts = {} (expected {};按 stage 分组计数,"
          "batch_train_ 批命名空间已剔除)".format(
              stage_fact_counts["decode"], manifest_count))
    check("R0c6", not cpp_duplicate_pairs,
          "C++ fact duplicated (request, stage 恰一次): {!r}".format(
              cpp_duplicate_pairs[:5]))
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
    #       真实请求对 + batch_train_ 哨兵/唤醒 watch 对(批命名空间
    #       watch 与请求级 watch 同计数器,双口径合并对平,M8-B2/E1
    #       修复 2026-09-27)== ledger stage 对计数 + 哨兵/唤醒对 ==
    #       sum(digests.watch_count);
    #   R2b 批级摘要合法:watch 每批 ∈ [0, node_count] 且与 ranks 空性一致、
    #   ranks ⊆ [0,53]、
    #       sum(node_count) == C++ total_nodes;
    #   R2c watch 生命周期干净收尾:phase-4 end audit 全零。
    watch_total_bad = []
    ledger_pairs = set()
    for rid, layers in python_ledger.items():
        for stage in ((layers.get("completed_unreconciled") or {})
                      .get("stages") or []):
            ledger_pairs.add((rid, stage["stage"]))
    # completed_nodes 只含请求域的 prefill/decode 数据节点(批命名空间已在
    # load_committed 剔除,request_complete 控制组无 completed_nodes)。
    committed_pairs = {key for key in committed_counts
                       if key[1] in (STAGE_PREFILL, STAGE_DECODE)}
    watch_facts = len(committed_pairs) + len(batch_train_watch_pairs)
    if cpp_audit is not None:
        if cpp_audit["total_watches"] != watch_facts:
            watch_total_bad.append({
                "cpp_total_watches": cpp_audit["total_watches"],
                "completed_nodes_pairs": len(committed_pairs),
                "batch_train_watch_pairs": len(batch_train_watch_pairs)})
        if cpp_audit["total_watches"] != len(ledger_pairs) \
                + len(batch_train_watch_pairs):
            watch_total_bad.append({
                "cpp_total_watches": cpp_audit["total_watches"],
                "ledger_stage_pairs": len(ledger_pairs),
                "batch_train_watch_pairs": len(batch_train_watch_pairs)})
        if cpp_audit["total_watches"] != watch_total:
            watch_total_bad.append({
                "cpp_total_watches": cpp_audit["total_watches"],
                "digest_watch_sum": watch_total})
    batch_watch_bad = []
    batch_rank_bad = []
    for digest in digests:
        wc = digest.get("watch_count", 0)
        ranks = digest.get("ranks") or []
        node_count = digest.get("node_count", 0)
        # 上界 = 批节点总数(同批多成员同 rank 退出时 watch 可超 rank 数;
        # 旧界 [0, len(ranks)] 在 ≥7 成员同批退出时误 FAIL)。
        if not (isinstance(wc, int) and 0 <= wc <= node_count
                and (wc == 0) == (len(ranks) == 0)):
            batch_watch_bad.append({
                "delivery_sequence": digest["delivery_sequence"],
                "watch_count": wc, "ranks_len": len(ranks),
                "node_count": node_count})
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
    audit_report_only = []
    if cpp_audit is not None:
        for field in ("watch_registry_size", "watch_stale",
                      "affected_ranks_residual"):
            if cpp_audit.get(field, 0) != 0:
                audit_bad.append({field: cpp_audit[field]})
        # completed_facts_residual is REPORT-ONLY by C++ contract: the
        # phase-4 end-audit comment in main_online.cc registers the
        # run-end tail facts (the final requests' end-barrier control
        # nodes complete AFTER the final delivery, phase-3 R5) as
        # "reported, not asserted" -- the residual is STRUCTURALLY
        # non-zero in the exact mode this tool itself requires, so the
        # old ==0 hard assertion failed every official recipe by
        # construction. It is carried into the report evidence instead.
        if cpp_audit.get("completed_facts_residual", 0) != 0:
            audit_report_only.append(
                {"completed_facts_residual":
                 cpp_audit["completed_facts_residual"]})
    check("R2a", not watch_total_bad,
          "watch 注册总数失配(每 (request, stage) 恰一次): {}".format(
              watch_total_bad))
    check("R2b", not batch_watch_bad and not batch_rank_bad
          and not node_total_bad,
          "批级 committed 摘要失配: watch 每批 ∈[0,node_count] 且空性一致 {} 条 / ranks 越界 "
          "{} 条 / node_count 总量 {} 条".format(
              len(batch_watch_bad), len(batch_rank_bad),
              len(node_total_bad)))
    check("R2c", not audit_bad,
          "watch 生命周期残留(phase-4 end audit 非零): {}".format(
              audit_bad))
    # completed_facts_residual 的 C++ 契约口径是"上报不断言"(见 R2c 上方
    # 注):仅在报告证据里携带,不参与判定。
    if audit_report_only:
        report_facts.append(
            "R2c-report-only: completed_facts_residual(运行尾差,C++ 契约"
            " reported-not-asserted) = {}".format(audit_report_only))

    # ---- R3 KV 账本对平(phase-7 契约) ----
    # kv_actions 随 response 文件消费即删,逐事件明细不再落盘;KV 生命周期
    # 权威迁移到:(a) online_decision_log.jsonl 的 completion 决策行(每
    # request 恰一条,kind_counts 校验;tick = KV 生命周期终止时刻)、
    # (b) cpp.log phase-5 total_kv_actions(每 request ≥ 1 动作的计数级
    # 保证)。
    # M32 修复:"每 request 恰一条"现在真实执行——重复行由 load_decision_log
    # 检出(decision_duplicates),kind 行数/唯一 request 数与 manifest 总数
    # 对平(decision_kind_counts 即该对账的自然消费点,原为只存不读死变量)。
    kv_kind_bad = []
    for kind in ("prefill", "decode", "completion"):
        kind_rows = decision_kind_counts.get(kind, 0)
        unique_requests = len(decision_log[kind])
        if kind_rows != manifest_count or unique_requests != manifest_count:
            kv_kind_bad.append({
                "kind": kind, "decision_rows": kind_rows,
                "unique_requests": unique_requests,
                "expected": manifest_count})
    kv_missing_bad = []
    kv_tick_bad = []
    kv_order_bad = []
    kv_instance_bad = []
    # (2026-10-01, D1 复审) 防御性默认:R3 行渲染无条件引用这两个列表。
    kv_low = []
    active_left = []
    # (2026-10-01) 条件收紧为 decision_rows 单检:load_decision_log 恒返
    # dict(无 None 返回路径),入口自检已保证 decision_rows > 0,原
    # `is not None` 臂与下方 else(报告 R3a FAIL)均不可达——入口 exit 2
    # 已覆盖该数据源缺失场景。
    if decision_rows:
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
            # B2 三态:REMOTE 态会话无本地实例(instance=None 合法,远端
            # 账面/终局核销口径);数值实例仍须落在 rank 域 [0,53]。
            if instance is not None and not (0 <= instance <= 53):
                kv_instance_bad.append({
                    "request_id": request_id,
                    "kv_instance_after_completion": instance})
        if cpp_audit is not None and cpp_audit["total_kv_actions"] \
                < len(manifest_ids):
            kv_low.append(cpp_audit["total_kv_actions"])
        if cpp_counters is not None and cpp_counters["active"] != 0:
            active_left.append(cpp_counters["active"])
        check("R3a", not kv_missing_bad and not kv_low
              and not decision_duplicates and not kv_kind_bad,
              "KV 生命周期缺失/重复/计数失配(每 request 恰 prefill/decode/"
              "completion 三条决策;C++ total_kv_actions >= 请求数): 缺 "
              "completion {} 条 / 重复决策 {} 条 / kind 计数失配 {} / "
              "动作计数过低 {}".format(
                  len(kv_missing_bad), decision_duplicates[:5],
                  kv_kind_bad, kv_low))
        check("R3b", not kv_tick_bad,
              "completion 决策 tick != ledger completed_tick: {} 条".format(
                  len(kv_tick_bad)))
        check("R3c", not kv_order_bad and not kv_instance_bad
              and not active_left,
              "KV 顺序/实例/收尾违例: 顺序 {} 条 / 实例越界 {} 条 / "
              "active 残留 {}".format(
                  len(kv_order_bad), len(kv_instance_bad), active_left))

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
    r4c_ok = check("R4c", final_queued.get("admitted_count") == 0,
                  "final decision boundary admitted_count = {} (expected 0)".format(
                      final_queued.get("admitted_count")))

    # ---- R5 injected-unfinished 对平 ----
    if query_log:
        last_row = query_log[-1]
        residuals = [
            entry for entry in last_row.get("injected_unfinished", [])
            if entry.get("node_count", 0) != 0]
        # 末决策边界残差 = 最后完成动作的 end-barrier 控制尾节点(在最终
        # watch 命中时仍在途,于运行结束前完成——C++ 干净退出 active=0
        # 即证)。列车化口径(2026-10-01):decode 的 end barrier 是列车
        # 共享节点(batch_train_ 批命名空间),残差组不再按命名空间裁剪,
        # 统一按 barrier 签名断言——decode 阶段、单节点、零 compute、零
        # 剩余时长、comm_bytes 为控制字量级(请求级尾 1/31 字节与列车
        # 共享尾 8 字节都在案,实体工作的 comm 是 MB-GB 量级,阈值取 64
        # 区分两界)。列车体/哨兵(多节点或 prefill 阶段)出现在末边界
        # 即真异常:运行不可能带着未完成的实体工作或未核销的截断哨兵
        # 干净退出。残差非空断言保留:全部排空 = 终动作 barrier 在摘要
        # 拍取前已完成,相位证据丢失。
        residual_bad = []
        residual_groups = []
        for entry in residuals:
            for group in entry.get("per_request", []):
                residual_groups.append(group)
        for group in residual_groups:
            signature_ok = (
                group.get("stage") == STAGE_DECODE
                and group.get("node_count") == 1
                and group.get("compute_ops") == 0
                and group.get("estimated_remaining_ns") == 0
                and group.get("comm_bytes", 0) <= 64)
            if not signature_ok:
                residual_bad.append(group)
        if not residual_groups:
            check("R5a", False,
                  "final epoch injected-unfinished fully drained "
                  "(expected the final completion's end-barrier residual)")
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
                    # batch_train_ 哨兵/唤醒标记节点走批命名空间,非请求级
                    # 注入账,无 committed 对可对平(M8-B2 前缀剥离同口径)。
                    if group["request_id"].startswith(BATCH_TRAIN_PREFIX):
                        continue
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

    # ---- R7 issued 层对平(阶段 7 §10.1,核心;列车感知口径) ----
    # Python issued(请求级:加入列车时登记、REQUEST_COMPLETE 核销于
    # 策略完成处理前)的物理在飞证据在两个命名空间之一:请求级节点
    # (joiner 迁移/exit 标记——prefill 段与 exit 标记在途窗口落请求级
    # 键),或列车体节点(批命名空间 train_id 键——列车体运行期间成员
    # 请求的在飞工作全部归属列车,成员关系由 train_ledger 回放)。R7a1
    # 断言每边界 issued 请求至少被其一覆盖;两侧时刻对齐:列车在其发射
    # 边界的 tick 末 deferred 提交,下一边界起进入 C++ 摘要,而 Python
    # 侧 join 登记同样发生在发射边界策略段(快照之后)——首个可见边界
    # 两侧同为下一边界。
    # 反向残差 only_cpp 允许为"end-barrier 控制尾节点"已知窗口:watch
    # 命中即报告阶段完成(Python 已核销 issued),但控制尾节点仍在途
    # (阶段 3 R5 同源语义)——请求级残差逐组核对签名(单节点、无
    # compute、无时长);batch_train 组(列车哨兵/唤醒/批级 barrier,
    # Python issued 是请求级键集,批命名空间组恒落 only_cpp 窗口且节点
    # 数是整列列车体)不参与该方向核对;末边界 issued 清空。
    issued_only_python = []
    issued_residual_bad = []
    residual_peak = 0
    for row in query_log:
        py_issued = {
            entry["request_id"]
            for entry in (row.get("issued") or {}).get("detail", [])
        }
        cc_keys = set()
        cc_train_ids = set()
        cc_groups = []
        for rank_entry in row.get("injected_unfinished", []):
            for group in rank_entry.get("per_request", []):
                cc_keys.add(group["request_id"])
                cc_groups.append(group)
                if group["request_id"].startswith(BATCH_TRAIN_PREFIX):
                    cc_train_ids.add(group["request_id"])
        only_python = set()
        for request_id in py_issued - cc_keys:
            if not any(train["train_id"] in cc_train_ids
                       for train in trains_of_request.get(request_id, ())):
                only_python.add(request_id)
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
            # batch_train 裁剪(3.1 修复,2026-09-29;与 R5b 同口径):
            # 批命名空间组(列车哨兵/唤醒/批级 barrier)不是请求级 issued
            # 账,Python 侧恒已剥离前缀结算,它们天然落在 only_cpp 窗口且
            # 节点数是整列列车体(>1)——未裁剪时被误判为 non-barrier 残差
            # (exact 模式实测 96 组),配方按构造失败。请求级残差仍逐组
            # 核对签名。
            if group.get("request_id", "").startswith(BATCH_TRAIN_PREFIX):
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
          "{} boundaries with Python-issued requests carrying neither "
          "request-level nor in-flight-train C++ evidence (issued ⊆ "
          "request-level keys ∪ live-train membership required)".format(
              len(issued_only_python)))
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
    # 中间节点,段内依赖链正常形态);列车可见性(列车化口径):每个
    # 当前列车(发射于本边界 tick 前、同实例后继列车尚未发射)只要有
    # 仍在 issued 的成员,就必须在 C++ 摘要出现——列车体/哨兵/共享
    # barrier 至少一个节点未完成;后继已发射的边界不再要求前车(交接
    # 边界由后车接棒,前车尾部 barrier 的在途窗口由 R7a2 签名核对覆
    # 盖)。原"在飞节点数 >= issued request 数"的数值口径不适应列车
    # 聚合:多个 issued 请求共享一趟列车的节点账,逐请求数值下界在
    # 列车世界无定义;峰值统计供报告列示。
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
        boundary_tick = row.get("tick", 0)
        py_issued = {
            entry["request_id"]
            for entry in (row.get("issued") or {}).get("detail", [])
        }
        cc_train_ids = {
            group["request_id"]
            for rank_entry in row.get("injected_unfinished", [])
            for group in rank_entry.get("per_request", [])
            if group["request_id"].startswith(BATCH_TRAIN_PREFIX)
        }
        for train in trains_by_emission:
            if train["tick"] >= boundary_tick:
                continue
            successor_tick = train["successor_tick"]
            if successor_tick is not None and successor_tick <= boundary_tick:
                continue
            issued_members = train["members"] & py_issued
            if issued_members and train["train_id"] not in cc_train_ids:
                cc_cover_bad.append({
                    "delivery_sequence": row["delivery_sequence"],
                    "train_id": train["train_id"],
                    "issued_members": sorted(issued_members)[:5],
                })
    check("R8a", not cc_internal_bad,
          "{} boundary summary inconsistencies (node_count >= in_flight + "
          "free)".format(len(cc_internal_bad)))
    check("R8b", not cc_cover_bad,
          "{} boundaries where a live train with issued members is absent "
          "from the C++ summary".format(len(cc_cover_bad)))

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
                kv_kind_bad=kv_kind_bad,
                decision_duplicates=decision_duplicates,
                batch_train_watch_pairs=len(batch_train_watch_pairs),
                kv_low=kv_low, active_left=active_left,
                admitted_left=admitted_left, committed_left=committed_left,
                ready_bad=ready_bad, final_ready=final_ready,
                issued_only_python=issued_only_python,
                issued_residual_bad=issued_residual_bad,
                residual_peak=residual_peak, final_issued=final_issued,
                cc_internal_bad=cc_internal_bad, cc_cover_bad=cc_cover_bad,
                peak_node=peak_node, peak_in_flight=peak_in_flight,
                peak_free=peak_free,
                stage_fact_counts=stage_fact_counts, r4c_ok=r4c_ok,
                report_facts=report_facts))
        print("[ledger_reconcile] report written: {}".format(args.report))
    print("[ledger_reconcile] verdict: {}".format(
        "对平 (balanced)" if balanced else "不平 (unbalanced)"))
    for fact in report_facts:
        print("[ledger_reconcile] REPORT-ONLY: {}".format(fact))
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
                   kv_instance_bad, kv_kind_bad, decision_duplicates,
                   batch_train_watch_pairs, kv_low, active_left,
                   admitted_left, committed_left,
                   ready_bad, final_ready, issued_only_python,
                   issued_residual_bad, residual_peak, final_issued,
                   cc_internal_bad, cc_cover_bad,
                   peak_node, peak_in_flight, peak_free,
                   stage_fact_counts, r4c_ok,
                   report_facts=None):
    report_facts = report_facts or []
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
    # R0/R5 行按本项(check id 前缀)结果渲染(py-verify-C1 修复:此前
    # 二者借用全局 balanced/!failures——任何其他项失败时误显 FAIL,单方向
    # 假 FAIL;退出码不受影响)。
    r0_bad = sorted({item for item, _ in failures if item.startswith("R0")})
    r5_bad = sorted({item for item, _ in failures if item.startswith("R5")})
    lines.append("| R0 集合对平(manifest / Python completed-unreconciled / C++ REQUEST_COMPLETE / cpp.log completed = {};prefill/decode 完成事实各 {}(按 stage 分组计数,批命名空间已剔除);零重复) | {} |".format(
        manifest_count, "{} prefill / {} decode".format(
            stage_fact_counts["prefill"], stage_fact_counts["decode"]),
        "PASS" if not r0_bad
        else "FAIL: {} 项失配({})".format(len(r0_bad), ",".join(r0_bad))))
    lines.append("| R1 逐 request 生命周期对平(admitted<=commit<=prefill<=decode==complete;gen 0/1/1) | {} |".format(
        "PASS" if not lifecycle_bad else "FAIL: {} 条".format(len(lifecycle_bad))))
    lines.append("| R2 committed 节点对平(phase-7:每 (request, stage) 恰注册一次 watch;真实请求 completed_nodes 对 + batch_train_ 哨兵/唤醒对 == ledger 对 + 哨兵/唤醒对 == digest watch 和 == cpp total_watches(双口径合并对平);批级摘要 watch_count∈[0,node_count] 且与 ranks 空性一致;rank ⊆ [0,53];digest node 和 == cpp total_nodes;phase-4 审计全零) | {} |".format(
        "PASS" if not (watch_total_bad or batch_watch_bad
                       or batch_rank_bad or node_total_bad or audit_bad)
        else "FAIL: watch 总数 {} 条 / 批 watch {} 条 / 批 rank {} 条 / node 总数 {} 条 / 审计 {} 条".format(
            len(watch_total_bad), len(batch_watch_bad),
            len(batch_rank_bad), len(node_total_bad), len(audit_bad))))
    lines.append("| R3 KV 账本对平(decision_log 每 request 恰一次 completion 决策,零重复、prefill/decode/completion 行数各= 请求数;completion tick == ledger completed_tick;prefill<=decode<=completion;kv 实例 id ∈ [0,53] 或 None(B2 三态 REMOTE 合法);cpp total_kv_actions >= 请求数;运行结束 active=0) | {} |".format(
        "PASS" if not (kv_missing_bad or kv_tick_bad or kv_order_bad
                       or kv_instance_bad or kv_kind_bad
                       or decision_duplicates or kv_low or active_left)
        else "FAIL: 缺 completion={} tick={} 乱序={} 实例={} 重复决策={} kind计数失配={} kv_low={} active_left={}".format(
            len(kv_missing_bad), len(kv_tick_bad), len(kv_order_bad),
            len(kv_instance_bad), len(decision_duplicates), kv_kind_bad,
            kv_low, active_left)))
    lines.append("| R4 排队账本对平(运行结束 admitted/committed 清空;末边界 admitted_count=0) | {} |".format(
        "PASS" if not (admitted_left or committed_left) and r4c_ok else
        "FAIL: admitted={} committed={}{}".format(
            len(admitted_left), len(committed_left),
            "" if r4c_ok else ";末边界 admitted_count!=0")))
    lines.append("| R5 injected-unfinished 对平(末决策边界残差 = 终请求 decode end-barrier 控制节点(列车化口径含列车共享 barrier);注入峰值 ≤ committed 节点数) | {} |".format(
        "PASS" if not r5_bad
        else "FAIL: {} 项失配({})".format(len(r5_bad), ",".join(r5_bad))))
    lines.append("| R6 ready 层对平(§10.1:每边界 ready ⊆ admitted;ready ∩ issued == ∅;末边界 ready=0) | {} |".format(
        "PASS" if not ready_bad and final_ready.get("ready_count", 0) == 0
        else "FAIL: 边界违例 {} 条 / 末边界 ready={}".format(
            len(ready_bad), final_ready.get("ready_count", 0))))
    lines.append("| R7 issued 层对平(§10.1 列车感知:每边界 Python issued 请求有请求级或所在列车的 C++ 在飞证据;only-C++ 请求级残差全部为 end-barrier 控制尾节点签名;末边界 issued=0) | {} |".format(
        "PASS" if not issued_only_python and not issued_residual_bad
        and final_issued.get("issued_count", 0) == 0
        else "FAIL: only-python {} 条 / 残差签名 {} 条 / 末边界 issued={} / 残差峰值 {}".format(
            len(issued_only_python), len(issued_residual_bad),
            final_issued.get("issued_count", 0), residual_peak)))
    lines.append("| R8 network pending/active 计数对账(§10.1:摘要内部一致 node_count ≥ in_flight+free;当前列车含 issued 成员即在摘要可见) | {} |".format(
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
        lines.append("- batch_train_ 哨兵/唤醒 watch 完成组对: {}(批命名空间,与真实请求对合并对平 cpp total_watches)".format(
            batch_train_watch_pairs))
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
    lines.append("| remote 池传输(远端共享 KV 池,remote_store/remote_load 经边缘流体端口) | 无逐请求 FIFO job 账本;逐事务事实 = remote_memory_transactions.jsonl,远端账面单列(py-metrics-C2:措辞按本仓机制改写) | 显式占位,明细即权威 |")
    lines.append("| network pending/active | C++ 执行事实(injected-unfinished 摘要) | 对平 R8(计数/审计) |")
    lines.append("| local HBM job | 无 local-HBM job 账本(计时模型已激活:"
                 "comm 边端点计费 hbm_charge 在案,裁决 #35;缺的是独立 "
                 "local-HBM 节点账本) | 显式占位,无账本 |")
    lines.append("| completed-unreconciled | Python 常驻 | 对平 R0 |")
    lines.append("")
    if report_facts:
        lines.append("## 上报不断言的运行尾差(C++ 契约 reported-not-asserted)")
        lines.append("")
        for fact in report_facts:
            lines.append("- {}".format(fact))
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
                 "<run>/cpp.log --report <out>.md`;退出 0 = 对平。对账数据源"
                 "要求运行侧 env SH_GRAPH_DIGESTS=1 与 "
                 "ASTRA_SIM_ONLINE_COMPLETED_NODES=exact(缺失即入口 "
                 "fail-closed 退出 2,消息内注明)。")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
