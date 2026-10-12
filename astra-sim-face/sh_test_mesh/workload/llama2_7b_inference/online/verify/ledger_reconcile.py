#!/usr/bin/env python3
"""ledger_reconcile.py -- 结束总账核对(方案 §6.2 操作 3 / contract ⑥;
阶段 7 §10.1 扩展为逐层对账)。

Python KV/排队账本 vs C++ 执行事实,逐 request/stage 对平。输入(同一
strategy 感知运行目录):

  --bridge-dir <run>/bridge   request_*.json(C++ 完成事实)
                              response_*.json(committed 节点 + kv_actions
                              + assignments)
                              graph_batch_digests.jsonl(批级 committed
                              摘要:watch_count / ranks / node_count)
                              ledger.jsonl(Python 分层账本导出)
                              sensing_query_log.jsonl(决策边界两层剩余负载
                              查询快照)
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
  R2 committed 节点对平(watch 成员账本):每 (request, stage) 恰注册一次
     watch,watch 成员数 == C++ completed_groups node_count(request 文件
     vs response 文件对平);watch 成员 rank ⊆ 该 (request, stage)
     committed 节点 rank 集。watch 成员 = 每 rank 最后一个该阶段节点
     (非全部阶段节点;阶段图可横跨 prefill+decode 两实例 rank),故
     committed rank 数可以大于成员数,不构成失配。
     R2b 加固(2026-08-16,wscllm 054118e 同款不变量;2026-09-29 R3-K3
     上界改列车化口径):graph_batch_ digests.jsonl 批级摘要合法——
     每批 watch_count ∈ [0, node_count] 且 watch_count==0 ⇔ ranks 为空
     (空性等价;列车化后同批含逐退出成员 exit watch,数量不受 rank 数
     约束,故上界锚定本批 committed 节点数——每 watch 的成员集非空且
     同批各 watch 成员集两两不交)、ranks ⊆ [0,53]、
     sum(node_count) == C++ phase-5 total_nodes 且 sum(watch_count)
     == phase-5 total_watches;R2c:phase-4 end audit 全零(watch 生命
     周期干净收尾)。
  R3 KV 账本对平(Python KV 权威 vs C++ 完成事实):每 request 至少 1 条
     kv_action(trigger_request_id == R);"completion" phase 事件
     (retain_complete)的 planner_time_ns == completed_tick;所有 kv 事件
     tick <= completed_tick。
  R4 排队账本对平:运行结束 admitted/committed 层已清空(ledger.jsonl 无
     admitted/committed 条目);最后一决策边界 admitted_count == 0。
  R5 injected-unfinished 对平(sensing_query_log):末决策边界残差 = 终列车
     decode end-barrier 控制节点(每 rank 1 个 all_reduce,无 compute、无
     时长;最终 watch 命中时仍在途,运行结束前完成——C++ 干净退出
     active=0 即证),断言残差非空且全部为 barrier 签名(comm_bytes ==
     该 rank 已提交 end-barrier 节点自身的 coll.bytes = 该列车 iterations;
     2026-09-30 口径修正:锚定对象从"该键全部已提交 collective 之和"改为
     "按名锚定 _end_barrier 节点自身字节"——列车键下还挂着列车体激活
     all_reduce(实测 512KiB/迭代),旧求和锚在末列车 1 迭代收尾车上报
     524288+1 vs 残差 1 的口径错位恒假 FAIL);注入峰值(节点数,跨
     rank/代求和)≤ committed 节点数(提交后逐批流入,完成时归零)。
  R6 ready 层对平(§10.1):每边界 ready(就绪可服务排队成员,策略 override
     视图)⊆ admitted(键集)且 ready ∩ issued == ∅;末决策边界 ready==0。
  R7 issued 层对平(§10.1,核心;2026-09-30 窗口口径修正):真实请求方向
     为 C++ injected-unfinished per_request 键集 ⊆ 同边界 Python issued
     键集——Python issued 是请求级"发射→完成核销"生命周期窗口(登记于
     emit 段、核销于 _settle_completions),C++ 摘要是 chunk 级瞬态窗口
     (列车化后每 chunk 提交后至下一快照前完成即永不出现,实测 21 请求
     中 5 个全程零出现、窗口覆盖 36/1294 边界),旧"Python ⊆ C++"逐边
     界前提在此形态下结构性恒假(1267/1294 边界 only-python);反向包含
     结构成立:C++ 侧出现未完成节点 ⟹ 节点已提交(发射于更早边界,Python
     已登记)且完成事实未核销(核销边界快照已不含)。only-C++ 残差按键
     类分流:真实请求残差必须为 end-barrier 控制尾节点签名(watch 命中
     即报阶段完成、控制尾节点仍在途的已知窗口,与 R5 同源);batch_train_
     列车键不走 issued 层(列车机械以 train_ledger.jsonl 记账,发射/核销
     独立于请求 issued 账本),残差形状不设限,但须与 train_ledger 交叉
     对平:每个出现的列车键都有台账行、且首现 tick ≥ 台账发射 tick;
     末决策边界 issued==0。
  R8 network pending/active 计数对账(§10.1):C++ 摘要内部一致(每 rank
     node_count >= in_flight_node_count + free_node_count,剩余 = 依赖未
     满足的未发射中间节点);在飞节点数(跨 rank)>= issued request 数(每
     在飞 request 至少 1 节点);峰值统计列示。

八层账本口径(§10.1):admitted / committed / ready / issued / network
pending/active 参与对平;remote FIFO 与 local HBM 为"不适用"显式占位
(face 无远端内存、无 LocalHbmBandwidthModel),报告中列示不参与对平。

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
    同一 (request, stage, generation) 重复出现 = 失配。
    batch_train_ 前缀的列车哨兵/唤醒完成事实与 _settle_completions 同口径
    跳过(见 online_scheduler_base.py 的 BATCH_TRAIN_PREFIX):它们不是请求
    完成事实,混入会让 R0c/R0c1/R0c3/R0c5 的集合对平假失败。"""
    completions = {}
    arrival_epochs = 0
    for seq, req in iter_request_records(bridge_dir):
        arrival_epochs += len(req.get("arrivals") or [])
        for group in req.get("completed_groups") or []:
            request_id = group["request_id"]
            if request_id.startswith("batch_train_"):
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


def _collect_end_barrier_bytes(barrier_bytes, node):
    """按 (rank, request_id, stage) 累计已提交 end-barrier 节点自身的
    coll.bytes(节点名 _end_barrier 后缀锚定;列车化 A2-2 口径 bytes =
    该列车 iterations)。2026-09-30 口径修正(对 2026-09-28 B7 的再修正):
    R5a barrier 参照锚定对象从"该键全部已提交 collective 求和"收窄为
    end-barrier 节点自身字节——列车键下还挂着列车体激活 all_reduce
    (实测 512 KiB/span),旧求和锚把末列车 1 迭代收尾车的在途 barrier
    (1 字节)对到 524288+1 上,两套口径在末边界相遇即恒假 FAIL。"""
    if not str(node.get("name", "")).endswith("_end_barrier"):
        return
    coll = node.get("coll")
    nbytes = coll.get("bytes") if isinstance(coll, dict) else None
    if isinstance(nbytes, int) and nbytes > 0:
        key = (node.get("rank"), node.get("request_id", ""),
               node.get("stage", ""))
        barrier_bytes[key] = barrier_bytes.get(key, 0) + nbytes


def load_responses(bridge_dir, manifest_path=None):
    """从 response_*.json 收集:每 (request_id, stage) committed 节点的去重
    rank 集合与节点总数;全部 kv_actions;assignments 计数;注册过的 stage
    watch(members = {rank: node_id} 字典);按 (rank, request_id, stage)
    累计已提交 end-barrier 节点自身的 coll.bytes(R5a barrier 参照,
    2026-09-30 按名锚定 _end_barrier 后缀)。

    face 适配(阶段 7 §10.3 起 response 消费即删):bridge 无 response 文件
    时,改用确定性 Python 侧重建——request_*.json(C++ 实际交付的 delta
    序列)逐一喂给 FaceOnlineScheduler,重建每个 GraphBatch 的
    watches/kv_actions/assignments/节点(调度器为确定性函数,与在线运行
    交付一致;幂等 fixture 同机制验证)。需要 --manifest。"""
    committed_ranks = {}     # (request_id, stage) -> set(rank)
    committed_counts = {}    # (request_id, stage) -> 节点总数
    coll_bytes = {}          # (rank, request_id, stage) -> Σ end-barrier bytes
    watches = {}             # (request_id, stage) -> [watch, ...]
    kv_actions = []
    assignment_count = 0
    if not _seq_files(bridge_dir, "response_"):
        # 显式 if+raise 做 fail-closed 前置:python -O 下裸 assert 被剥
        # 掉,校验失效后只会以 open(None) 的 TypeError 混乱失败。
        if not manifest_path:
            raise AssertionError(
                "bridge 无 response_*.json(消费即删);对账需要 --manifest 做"
                "确定性 Python 侧重建")
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
        # M11/R5-S1(2026-09-29):确定性重放全程不喂 commit ack(重建完才
        # 结束),未确认交付数随全量输入增长;B4 delivery 类在途尾部默认
        # 上限 8 = 生产桥背压窗口口径,第 9 笔交付即 fail-closed RuntimeError
        # (对账工具在真实规模不可用)。构造前显式调大(官方 30s 输入
        # 1177 交付;环境变量 SH_PROPAGATING_TAIL_LIMIT 显式设置时仍最
        # 优先)。生产 online_service 路径不受影响。
        FaceOnlineScheduler.propagating_tail_delivery_limit = 4096
        with open(manifest_path, "r", encoding="utf-8") as _src:
            _manifest = json.load(_src)
        _config = load_face_trace_config(
            _Path(os.path.join(_wl, "trace_config.csv")))
        _sched = FaceOnlineScheduler(
            manifest=_manifest, config=_config,
            graph=GraphBatchBuilder(_config), mode="strategy", sensing=True)
        for _seq, _delta in iter_request_records(bridge_dir):
            _batch = _sched.on_decision_batch(_delta)
            for node in _batch.get("nodes") or []:
                key = (node.get("request_id", ""), node.get("stage", ""))
                committed_ranks.setdefault(key, set()).add(int(node["rank"]))
                committed_counts[key] = committed_counts.get(key, 0) + 1
                _collect_end_barrier_bytes(coll_bytes, node)
            for watch in _batch.get("watches") or []:
                key = (watch.get("request_id", ""), watch.get("stage", ""))
                watches.setdefault(key, []).append(watch)
            kv_actions.extend(_batch.get("kv_actions") or [])
            assignment_count += len(_batch.get("assignments") or [])
        return committed_ranks, committed_counts, watches, kv_actions, \
            assignment_count, coll_bytes
    for seq in _seq_files(bridge_dir, "response_"):
        path = os.path.join(bridge_dir, "response_{}.json".format(seq))
        with open(path, "r", encoding="utf-8") as source:
            resp = json.load(source)
        for node in resp.get("nodes") or []:
            key = (node.get("request_id", ""), node.get("stage", ""))
            committed_ranks.setdefault(key, set()).add(int(node["rank"]))
            committed_counts[key] = committed_counts.get(key, 0) + 1
            _collect_end_barrier_bytes(coll_bytes, node)
        for watch in resp.get("watches") or []:
            key = (watch.get("request_id", ""), watch.get("stage", ""))
            watches.setdefault(key, []).append(watch)
        kv_actions.extend(resp.get("kv_actions") or [])
        assignment_count += len(resp.get("assignments") or [])
    return committed_ranks, committed_counts, watches, kv_actions, \
        assignment_count, coll_bytes


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


def load_train_ledger(bridge_dir):
    """train_ledger.jsonl(列车机械台账,与列车发射同边界逐行写出;
    R7a3 数据源)。文件缺失(无列车运行的 sensing 跑)返回空表——
    此时 injected 摘要若仍出现 batch_train_ 键,R7a3 以无台账行判失配。"""
    path = _artifact_path(bridge_dir, "train_ledger.jsonl")
    if not os.path.isfile(path):
        return []
    return list(_iter_jsonl(path))


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
    2026-08-16 wscllm 054118e 同款)。该文件仅当生产端以 SH_GRAPH_DIGESTS=1
    运行时产出(online_service C2/D2(2026-08-28) 开关,默认 "0" 关:
    digest 行是纯审计产物且每批全量二次序列化,默认不写;产出时每交付
    批一行,bridge/ 直读)。默认官方运行无此文件 ⇒ 返回 None,由对账项
    R2b fail-closed(不允许静默跳过);需对平批级摘要时以 SH_GRAPH_DIGESTS=1
    重跑后再对账。"""
    path = _artifact_path(run_bridge_dir, "graph_batch_digests.jsonl")
    if not os.path.exists(path):
        return None
    rows = list(_iter_jsonl(path))
    return rows or None


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
             "跨输入复用时按物化实测改传,来源:face测试/开感知120报告"
             "_3mins.md 建议)")
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

    def row_int(source, key, *, item, where):
        """R9-F3(2026-09-29):摘要行级数值字段严格取值助手。

        旧 row.get(k, 0) 直接算术是 fail-open:缺列静默按 0 混入求和、
        float 静默截断、str 让求和点裸 TypeError 绕过本工具 check() 通道。
        此处缺列/类型错(bool 亦拒)记一条 <item> 失败并回退 0,同一数据
        源的其余违例仍可继续收集(对账报告一次列全)。"""
        if not isinstance(source, dict) or key not in source:
            check(item, False,
                  "{}: missing field {!r}".format(where, key))
            return 0
        value = source[key]
        if isinstance(value, bool) or not isinstance(value, int):
            check(item, False,
                  "{}: field {!r} must be int, got {!r}".format(
                      where, key, value))
            return 0
        return value

    # ---- R0 集合对平 ----
    manifest_ids = load_manifest(args.manifest)
    cpp_completions, arrival_epochs = load_cpp_facts(args.bridge_dir)
    committed_ranks, committed_counts, watches, kv_actions, \
        assignment_count, coll_bytes = load_responses(
            args.bridge_dir, manifest_path=args.manifest)
    python_ledger = load_python_ledger(args.bridge_dir)
    query_log = load_sensing_query_log(args.bridge_dir)
    cpp_counters = load_cpp_counters(args.cpp_log)
    digests = load_digests(args.bridge_dir)
    phase_counters = load_cpp_phase_counters(args.cpp_log)
    # (2026-10-01) --cpp-log 显式给出后的 fail-closed 加固(与
    # diff_explainability 的 H2/R3-K1 同款纪律):此前"文件不可读"与
    # "计数行解析不出"都静默返回 None,R0d/R0e 与 R2b digest 双和/R2c
    # 全部跳过——而日志截断/格式漂移恰是最需要对账把门的崩溃现场,
    # 假绿 exit 0 不可接受。未传 --cpp-log 仍是合法跳过(可选数据源)。
    if args.cpp_log is not None:
        if _read_cpp_log(args.cpp_log) is None:
            check("R0d", False,
                  "--cpp-log 指定但文件不可读(含 .gz 回退): {}".format(
                      args.cpp_log))
        else:
            if cpp_counters is None:
                check("R0d", False,
                      "--cpp-log 在场但 [online] service counters 行缺失/"
                      "不可解析(fail-closed,R0d/R0e 不再静默跳过)")
            if phase_counters["phase4"] is None:
                check("R2c", False,
                      "--cpp-log 在场但 phase-4 end audit 行缺失/不可解析"
                      "(fail-closed,R2c 不再静默跳过)")
            if (phase_counters["total_nodes"] is None or
                    phase_counters["total_watches"] is None):
                check("R2b", False,
                      "--cpp-log 在场但 phase-5 commit counters 行缺失/"
                      "不可解析,digest 双和无法对平(fail-closed)")

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
        rid: entry["prefill"] for rid, entry in cpp_completions.items()
        if entry.get("prefill") is not None}
    decode_facts = {
        rid: entry["decode"] for rid, entry in cpp_completions.items()
        if entry.get("decode") is not None}
    req_facts = {
        rid: entry["request_complete"]
        for rid, entry in cpp_completions.items()
        if entry.get("request_complete") is not None}
    # (2026-10-01) per-stage presence filter: the dicts used to keep
    # None-valued keys, so len() counted "requests with ANY fact" and a
    # systematically missing stage (e.g. no request_complete group anywhere)
    # still passed R0c/R0c1/R0c2 with a misleading evidence line. Downstream
    # readers use .get(rid) or {}, which treats an absent key exactly like
    # the old None value.
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

    # ---- R2b 批级摘要不变量(2026-08-16 加固,wscllm 054118e 同款) ----
    # 空性等价 watch_count==0 ⇔ ranks 空(同批多 watch 合法,禁 {0,1}
    # 假设);ranks ⊆ [0,53];digest 双和与 C++ phase-5 权威计数对平
    # (sum(node_count)==total_nodes、sum(watch_count)==total_watches)。
    # R3-K3(2026-09-29):上界改列车化口径 watch_count ≤ node_count——
    # 每 watch 的成员集非空且同批各 watch 成员集两两不交(drain 逐 drain
    # 成员、exit 逐退出成员、哨兵/唤醒各 ≤1),故 watch 数 ≤ 本批 committed
    # 节点数。旧逐请求时代口径 watch_count ≤ len(ranks) 在列车化后恒误
    # 报:同批 B 个 exit watch 遇 B-1 rank 即 B > len(ranks)(6 rank +
    # 8 成员实测复现假 FAIL)。行级数值经 row_int 严格取值(R9-F3)。
    digest_bad = []
    if digests is None:
        check("R2b", False,
              "graph_batch_digests.jsonl 缺失或 0 行(数据源缺失,fail-"
              "closed;不允许静默跳过批级摘要不变量)")
    else:
        node_sum = 0
        watch_sum = 0
        for row in digests:
            where = "digest delivery_sequence={!r}".format(
                row.get("delivery_sequence"))
            ranks = row.get("ranks")
            if not isinstance(ranks, list) or any(
                    isinstance(rank, bool) or not isinstance(rank, int)
                    for rank in ranks):
                check("R2b", False,
                      "{}: ranks 必须为 int 列表, got {!r}".format(
                          where, ranks))
                ranks = []
            watch_count = row_int(row, "watch_count", item="R2b", where=where)
            node_count = row_int(row, "node_count", item="R2b", where=where)
            node_sum += node_count
            watch_sum += watch_count
            seq = row.get("delivery_sequence")
            if not 0 <= watch_count <= node_count:
                digest_bad.append({
                    "delivery_sequence": seq,
                    "violation": "watch_count 不在 [0, node_count]"
                                 "(列车化上界)",
                    "watch_count": watch_count, "node_count": node_count})
            if (watch_count == 0) != (len(ranks) == 0):
                digest_bad.append({
                    "delivery_sequence": seq,
                    "violation": "空性等价失配(watch_count==0 ⇔ ranks 空)",
                    "watch_count": watch_count, "ranks": ranks})
            if any(not 0 <= rank <= 53 for rank in ranks):
                digest_bad.append({
                    "delivery_sequence": seq,
                    "violation": "rank 越界(ranks ⊄ [0,53])",
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
            max_tick = max(event["planner_time_ns"]
                           for event in completion_events)
            if max_tick != completed_tick:
                kv_tick_bad.append({
                    "request_id": request_id,
                    "completion_kv_tick": max_tick,
                    "completed_tick": completed_tick})
        if any(event["planner_time_ns"] > completed_tick
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
    # (2026-10-01 口径注记) R4a/R4b 在现行导出格式下是恒真防御门:
    # ledger.jsonl 每行固定只写 completed_unreconciled 层(生产侧
    # _ledger_complete),admitted/committed 层从不落盘。真正执法点是
    # 运行期 verify_run_end(sensing 分支)与 R4c(末边界 admitted_count,
    # 可判定)。两门保留作格式漂移哨兵:若未来 ledger 开始导出其它层,
    # 这里的"清空"断言立即生效。
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
        # 末决策边界残差 = 终列车 decode 的 end barrier 控制节点(每 rank
        # 1 个 all_reduce,无 compute、无时长模型;在最终 watch 命中时仍在
        # 途,于运行结束前完成——C++ 干净退出 active=0 即证)。列车化口径
        # 下 collective bytes = 列车 iterations(graph_batch_builder A2-2
        # 既定口径,末列车通常 ≠1),因此 R5a 断言:残差非空(终列车
        # barrier 必然在途)且全部为 barrier 签名(comm_bytes 与该 rank
        # 已提交 end-barrier 节点自身的 coll.bytes 一致;2026-09-28 B7 适配
        # 逐请求→列车口径,2026-09-30 再修正锚定对象:按名锚定
        # _end_barrier 节点字节,不再对该键全部已提交 collective 求和——
        # 求和锚混入列车体激活 all_reduce(512 KiB/span),末列车 1 迭代
        # 收尾车的 1 字节在途 barrier 会对上 524288+1 恒假 FAIL)。
        residual_bad = []
        residual_groups = []
        for entry in residuals:
            for group in entry.get("per_request", []):
                residual_groups.append((entry.get("rank"), group))
        for rank, group in residual_groups:
            barrier_bytes = coll_bytes.get(
                (rank, group.get("request_id"), group.get("stage")))
            # (2026-10-01) both stages are legal barrier residuals: face
            # allows P/D on different instances, so an instance's final train
            # can be a pure PREFILL train whose end barrier is still in
            # flight at the last delivery (the byte anchoring by
            # _end_barrier node name is what proves it IS a barrier). The
            # old stage=="decode" narrowing false-FAILED that shape.
            signature_ok = (
                (group.get("stage") == "decode" or
                 group.get("stage") == "prefill")
                and group.get("node_count") == 1
                and group.get("compute_ops") == 0
                and group.get("estimated_remaining_ns") == 0
                and barrier_bytes is not None
                and group.get("comm_bytes") == barrier_bytes)
            if not signature_ok:
                evidence = dict(group)
                if barrier_bytes is not None:
                    evidence["expected_barrier_bytes"] = barrier_bytes
                residual_bad.append(evidence)
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
            where = "query delivery_sequence={!r}".format(
                row.get("delivery_sequence"))
            epoch_totals = {}
            for rank_entry in row.get("injected_unfinished") or []:
                for group in rank_entry.get("per_request") or []:
                    request_id = group.get("request_id")
                    stage = group.get("stage")
                    node_count = row_int(group, "node_count", item="R5b",
                                         where=where)
                    if request_id is None or stage is None:
                        check("R5b", False,
                              "{}: per_request group missing request_id/"
                              "stage: {!r}".format(where, group))
                        continue
                    key = (request_id, stage)
                    epoch_totals[key] = (epoch_totals.get(key, 0)
                                         + node_count)
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
                "delivery_sequence": row.get("delivery_sequence"),
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

    # ---- R7 issued 层对平(阶段 7 §10.1,核心;2026-09-30 窗口口径修正) ----
    # 真实请求方向为 C++ ⊆ Python:Python issued 是请求级"发射→完成核销"
    # 生命周期窗口(emit 段登记、completion 核销于策略完成处理前,基类
    # _settle_completions);C++ 摘要是 chunk 级瞬态窗口(列车化后本批节点
    # 于快照后 commit、下一快照前即完成则永不出现,实测 21 请求中 5 个
    # 全程零出现)。旧"Python issued ⊆ C++ injected"逐边界前提在此形态下
    # 结构性恒假(2s 冒烟 1267/1294 边界 only-python),反向包含结构成立:
    # C++ 侧出现未完成节点 ⟹ 节点已提交(发射于更早边界,Python 已登记)
    # 且完成事实未核销(核销边界快照已不含)。
    # only-C++ 残差按键类分流(2026-10-01 注释修正:旧文描述的是
    # 2026-09-30 反向包含修正前的容忍语义)。现行口径:列车化后 barrier/
    # 列车体节点一律以 train_id 记账,真实请求键的节点在其 watch fire
    # 时已全部完成——真实请求出现在 only-C++ 窗口恒为 R7a1 违例;
    # R7a2 的残差签名循环只是 R7a1 的重复防御,不构成独立门。
    # batch_train_ 列车键不走 issued 层(列车机械以 train_ledger
    # 记账,发射/核销独立于请求 issued 账本),残差形状不设限,但须与
    # train_ledger 交叉对平(R7a3:有台账行、首现 tick ≥ 台账发射 tick)。
    # 末边界 issued 清空(R7b,issued 层漏核销的终态哨兵)。
    train_ledger_rows = load_train_ledger(args.bridge_dir)
    train_ledger_ticks = {}
    for ledger_row in train_ledger_rows:
        train_ledger_ticks[ledger_row["train_id"]] = min(
            train_ledger_ticks.get(ledger_row["train_id"],
                                   ledger_row["tick"]),
            ledger_row["tick"])
    issued_cc_outside_py = []
    issued_residual_bad = []
    train_unaccounted = []
    train_first_seen = {}
    residual_peak = 0
    for row in query_log:
        where = "query delivery_sequence={!r}".format(
            row.get("delivery_sequence"))
        py_issued = set()
        for entry in (row.get("issued") or {}).get("detail") or []:
            request_id = entry.get("request_id")
            if request_id is None:
                check("R7a1", False,
                      "{}: issued detail entry missing request_id: {!r}"
                      .format(where, entry))
                continue
            py_issued.add(request_id)
        cc_keys = set()
        cc_groups = []
        for rank_entry in row.get("injected_unfinished") or []:
            for group in rank_entry.get("per_request") or []:
                request_id = group.get("request_id")
                if request_id is None:
                    check("R7a2", False,
                          "{}: injected group missing request_id: {!r}"
                          .format(where, group))
                    continue
                cc_keys.add(request_id)
                cc_groups.append(group)
        only_cpp = cc_keys - py_issued
        residual_peak = max(residual_peak, len(only_cpp))
        cc_outside = {
            key for key in only_cpp if not key.startswith("batch_train_")}
        if cc_outside:
            issued_cc_outside_py.append({
                "delivery_sequence": row.get("delivery_sequence"),
                "cpp_outside_python_issued": sorted(cc_outside)[:5]})
        for key in only_cpp:
            if key.startswith("batch_train_"):
                first = train_first_seen.get(key)
                if first is None or row["tick"] < first[1]:
                    train_first_seen[key] = (
                        row.get("delivery_sequence"), row["tick"])
        # 残差核对:only_cpp 的真实请求组必须是 barrier 签名(控制尾节点);
        # 列车键形状不设限(列车体/屏障合法工作形态)。
        for group in cc_groups:
            if group["request_id"] not in only_cpp:
                continue
            if group["request_id"].startswith("batch_train_"):
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
    check("R7a1", not issued_cc_outside_py,
          "{} boundaries with C++ injected outside Python issued (real "
          "requests: C++ keys ⊆ Python issued required): {!r}".format(
              len(issued_cc_outside_py), issued_cc_outside_py[:3]))
    check("R7a2", not issued_residual_bad,
          "{} non-barrier residual groups in the only-C++ window (real "
          "requests: end-barrier control tails only)".format(
              len(issued_residual_bad)))
    for key, (seq, tick) in sorted(train_first_seen.items()):
        # (2026-10-01) SH_FIRST_TOKEN_SPLIT=1 的首步批唤醒标记是节点级
        # request_id "<train_id>_first_step"(graph_batch_builder 的
        # emit_train_first_step),而台账行以裸 train_id 为键(首步批在
        # 同一 id 上带 first_step 标记)——查找前剥离唤醒后缀,否则拆分
        # 模式的研究跑在 only-C++ 窗口出现唤醒标记时 R7a3 恒假 FAIL
        # ("无 train_ledger 行")。train_id 形如 batch_train_i{inst}_{seq}
        # (纯数字序号结尾),_first_step 后缀无歧义。
        ledger_key = (
            key[:-len("_first_step")]
            if key.endswith("_first_step") else key)
        emitted_tick = train_ledger_ticks.get(ledger_key)
        if emitted_tick is None:
            train_unaccounted.append({
                "train_id": key, "first_seen_delivery_sequence": seq,
                "violation": "无 train_ledger 行(未记账列车)"})
        elif tick < emitted_tick:
            train_unaccounted.append({
                "train_id": key, "first_seen_delivery_sequence": seq,
                "violation": "首现 tick {} < 台账发射 tick {}".format(
                    tick, emitted_tick)})
    check("R7a3", not train_unaccounted,
          "{} unaccounted train keys in the only-C++ window (train_ledger "
          "row required; first-seen tick >= emission tick): {!r}".format(
              len(train_unaccounted), train_unaccounted[:3]))
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
        where = "query delivery_sequence={!r}".format(
            row.get("delivery_sequence"))
        node_total = in_flight_total = free_total = 0
        for rank_entry in row.get("injected_unfinished") or []:
            # R9-F3(2026-09-29):行级计数严格取值(缺列/类型错走 check
            # 通道并按 0 继续),不再 row.get(k, 0) 裸算术。
            node_count = row_int(rank_entry, "node_count", item="R8a",
                                 where=where)
            in_flight = row_int(rank_entry, "in_flight_node_count",
                                item="R8a", where=where)
            free = row_int(rank_entry, "free_node_count", item="R8a",
                           where=where)
            node_total += node_count
            in_flight_total += in_flight
            free_total += free
            if node_count < in_flight + free:
                cc_internal_bad.append({
                    "delivery_sequence": row.get("delivery_sequence"),
                    "rank": rank_entry.get("rank"),
                    "node_count": node_count,
                    "in_flight": in_flight,
                    "free": free,
                })
        peak_node = max(peak_node, node_total)
        peak_in_flight = max(peak_in_flight, in_flight_total)
        peak_free = max(peak_free, free_total)
        issued_count = row_int((row.get("issued") or {}), "issued_count",
                               item="R8b", where=where)
        if issued_count and in_flight_total < issued_count:
            cc_cover_bad.append({
                "delivery_sequence": row.get("delivery_sequence"),
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
                admitted_left=admitted_left, committed_left=committed_left,
                ready_bad=ready_bad, final_ready=final_ready,
                final_queued=final_queued,
                issued_cc_outside_py=issued_cc_outside_py,
                issued_residual_bad=issued_residual_bad,
                train_unaccounted=train_unaccounted,
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
                   admitted_left, committed_left, ready_bad, final_ready,
                   final_queued,
                   issued_cc_outside_py, issued_residual_bad,
                   train_unaccounted, residual_peak,
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
    # (2026-10-01) 行级判据改局部:旧 R0/R5 行用全局 failures/balanced,
    # 其它项失败会把本项染红;R4 行漏 R4c(末边界 admitted_count),该门
    # 单独失败时本行仍显 PASS。全局判定与差异清单不受影响。
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
    lines.append("| R2b 批级摘要不变量(watch_count ∈ [0,node_count],R3-K3 列车化上界;空性等价 watch_count==0 ⇔ ranks 空;ranks ⊆ [0,53];digest 双和 == cpp phase-5 total_nodes/total_watches;2026-08-16 加固,wscllm 054118e 同款) | {} |".format(
        # (2026-10-01) digests is None = 数据源缺失走独立文案:digest_bad
        # 恒为 list(缺失时为空表),旧的三目把它渲染成 "FAIL: 0 条违例/
        # 失配","数据源缺失"分支实际不可达。
        "FAIL: 数据源缺失" if digest_bad is None else
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
    lines.append("| R4 排队账本对平(运行结束 admitted/committed 清空;末边界 admitted_count=0) | {} |".format(
        "PASS" if not (admitted_left or committed_left)
        and final_queued.get("admitted_count") == 0 else
        "FAIL: admitted={} committed={} 末边界 admitted_count={}".format(
            len(admitted_left), len(committed_left),
            final_queued.get("admitted_count"))))
    lines.append("| R5 injected-unfinished 对平(末决策边界残差 = 终列车 end-barrier 控制节点,bytes 按名锚定 _end_barrier 节点自身;注入峰值 ≤ committed 节点数) | {} |".format(
        "PASS" if not any(item.startswith("R5")
                          for item, _ in failures) else "FAIL"))
    lines.append("| R6 ready 层对平(§10.1:每边界 ready ⊆ admitted;ready ∩ issued == ∅;末边界 ready=0) | {} |".format(
        "PASS" if not ready_bad and final_ready.get("ready_count", 0) == 0
        else "FAIL: 边界违例 {} 条 / 末边界 ready={}".format(
            len(ready_bad), final_ready.get("ready_count", 0))))
    lines.append("| R7 issued 层对平(§10.1,2026-09-30 窗口口径修正:真实请求 C++ injected ⊆ 同边界 Python issued;真实请求 only-C++ 残差全部为 end-barrier 控制尾节点签名;列车键与 train_ledger 交叉对平;末边界 issued=0) | {} |".format(
        "PASS" if not issued_cc_outside_py and not issued_residual_bad
        and not train_unaccounted
        and final_issued.get("issued_count", 0) == 0
        else "FAIL: cpp-外溢 {} 条 / 残差签名 {} 条 / 未记账列车 {} 条 / 末边界 issued={} / 残差峰值 {}".format(
            len(issued_cc_outside_py), len(issued_residual_bad),
            len(train_unaccounted),
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
    lines.append("| remote-memory FIFO | 不适用(face 无远端内存后端 FIFO) | 显式占位,无账本 |")
    lines.append("| network pending/active | C++ 执行事实(injected-unfinished 摘要) | 对平 R8(计数/审计) |")
    lines.append("| local HBM job | 不适用(face 无 LocalHbmBandwidthModel,"
                 "全仓 grep 零命中) | 显式占位,无账本 |")
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
                 "<run>/cpp.log --expected-requests N "
                 "--expected-accepted-sessions N --report <out>.md`;"
                 "退出 0 = 对平;期望常量缺省 1177/112(30s 验收值)。")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
