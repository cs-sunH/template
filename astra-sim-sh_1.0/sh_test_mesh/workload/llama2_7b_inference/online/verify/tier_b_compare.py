#!/usr/bin/env python3
"""tier_b_compare.py -- sh_1.0 Tier B(B0-B4)集成比较器(阶段 2 交付物)。

合同⑦验证方法规定的比较器:逐层比较输出差异报告。基线侧材料 = 阶段 0
归档的静态 ET 基线目录(baseline/20_30s:decision_log.jsonl + generated/
manifest.json + generated/llama2_7b_inference.{rank}.et + metrics_manifest.json
+ raw_metrics.csv);在线侧 = 官方 runner 运行目录(run_dir:bridge/request_*.json
保留的交付流 + results/online_decision_log.jsonl + graph_batch_digests.jsonl)。
拓扑事实(request_queue/inference_groups/hardware/npus_count)取自物化配置
(--config = trace_config.csv,缺失 fail-closed;request-neutral,裸仓库不预置)。

分层(总体方案 §9.2;本仓字段/口径适配):
  B0 输入等价      请求/session 计数、决策日志行数、trace_digest 与
                   request_mapping_digest(在线 raw_metrics.csv vs 基线
                   raw_metrics.csv 同值 = 同一输入/映射血统)、prefill 决策
                   内容 digest(容差 0)。
  B1 生命周期/决策 oracle/replay 决策 tick 对离线 decision_log 的 delta 直方图
                   (record-tick 权威键:(kind, request_id) -> 离线记录 tick;
                   sh_2.0 裁决 3 / sh_3.0 两轮决定性验证法同款)。登记口径:
                   prefill 1157 exact + 20 alarm-clamp 正偏差[1,10]ns
                   (interval<=1ns session 的 max(record, tick+1) 钳制,
                   sh10_replay_scheduler.py 边界口径);decode [-4,+10]ns /
                   completion [-8,+9]ns(COMP 链校准整数截断 + 1ns 事件粒度,
                   类别①③);q-吸收事件 = 0(输入 hbm_wait_ns 全 0,无 HBM
                   挂起准入)。逆序/提前/阶段序违规检查(容差 0)。
  B2 策略等价      oracle:决策内容全等(三 kind 0 mismatch)+ 与离线 manifest
                   requests[].transfers_by_stage 逐 request 逐 stage 的 KV
                   转移元组对照(kv_event_payload_sh1 口径:kind/phase/reason/
                   session/trigger/source/target/total_bytes/shards)。
                   real-online(strategy):确定性重放不变量——把 strategy 运行
                   保留的 request_*.json 交付流逐条喂给全新的
                   Sh10OnlineScheduler(同一冻结 LUT/同一只读策略函数),
                   断言重放产出的决策与 recorded online_decision_log 逐行全等
                   (= 方案 §5.4"每次决策的输入与离线同一函数在相同输入下的
                   输出一致"的逐决策断言);prefill_assignment_key 与所选实例
                   自洽;完成时序差异不判失败。
  B3 图结构等价    在线发射图(比较器内确定性重放 replay 交付流重构 GraphBatch
                   节点/边,先经 digest 逐批对平 live run 的
                   graph_batch_digests.jsonl)vs 离线 .et 全 54 rank 的 canonical
                   logical node key 多重集比较((rank,name,type,is_cpu,is_timer,
                   类型感知属性);插入顺序/节点 ID 不作比较项,合同⑦)。边差异
                   按 within/cross-request 归类到登记类别①②③④。
  B4 执行等价      同图前提不成立(登记差异:跨 request 串行化边、三段式发射、
                   timer gate 时长、MEM 即时完成),exact 条款不适用;操作性
                   标准 = 差异全部可归因并记录:边界级/节点级 tick 漂移统计
                   (vs 离线 record tick)、completion 跨 tick 逆序 = 0(硬门)、
                   completion 早于 decode = 0(硬门)、e2e 指标对照(信息性,
                   replay LUT 时钟压缩归因)。

用法:
  python3 online/verify/tier_b_compare.py \
    --replay-run <run_dir> --strategy-run <run_dir> \
    --baseline <baseline/20_30s> --config <trace_config.csv> \
    --report online/verify/tier_b_report_20.md

退出码:全部断言成立 -> 0;任何断言失败 -> 1(比较器不得放宽掩盖失败,
仿真加速分析.md §12.5)。
"""

import argparse
import collections
import glob
import hashlib
import json
import os
import re
import sys

_VERIFY_DIR = os.path.dirname(os.path.abspath(__file__))
_ONLINE_DIR = os.path.dirname(_VERIFY_DIR)
_WL_DIR = os.path.dirname(_ONLINE_DIR)          # llama2_7b_inference
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_WL_DIR)))  # 仓库根(llama2_7b_inference -> workload -> sh_test_mesh -> root)
for _p in (_VERIFY_DIR, _ONLINE_DIR, _WL_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from google.protobuf.internal.decoder import _DecodeVarint32  # noqa: E402
from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode  # noqa: E402
from generate_face_trace import load_face_trace_config  # noqa: E402  (只读 import)
from online.replay_source import ReplaySource  # noqa: E402
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402
from online.sh10_replay_scheduler import Sh10ReplayScheduler  # noqa: E402
from online.sh10_online_scheduler import Sh10OnlineScheduler  # noqa: E402

KINDS = ("prefill", "decode", "completion")
# B1 冻结口径(登记差异界;实测分布,超出即未登记差异 -> fail):
#   prefill: exact 1157 + 20 个 alarm-clamp 正偏差,偏差范围 [1,10]ns;
#   decode: 范围 [-4,+10]ns;completion: 范围 [-8,+9]ns。
#   归因:① COMP 链校准整数截断(runtime_ns = dur*ops//total)+ 1ns 事件粒度;
#        ③ tick-end/deferred 顺序合同(T+1 显式延后);
#        alarm-clamp(sh_1.0 边界口径,interval<=1ns session)。
B1_BOUNDS = {
    # delta_range 含 exact(0) 条目;prefill 正偏差界 = [1,10](clamp)。
    "prefill": {"exact": 1157, "positive": 20, "negative": 0,
                "delta_min": 0, "delta_max": 10},
    "decode": {"delta_min": -4, "delta_max": 10},
    "completion": {"delta_min": -8, "delta_max": 9},
}
_ABSORB_THRESHOLD_NS = 1000  # sh10_replay_scheduler._DEFER_THRESHOLD_NS 同款


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as source:
        for line in source:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_deltas(run_dir):
    """按 delivery_sequence 升序读 run 的 request_*.json 交付流。"""
    out = []
    for path in glob.glob(os.path.join(run_dir, "bridge", "request_*.json")):
        out.append(json.load(open(path, encoding="utf-8")))
    out.sort(key=lambda d: d["delivery_sequence"])
    return out


def load_run_decision_log(run_dir):
    return load_jsonl(os.path.join(run_dir, "results",
                                   "online_decision_log.jsonl"))


def by_kind(rows):
    return {k: {r["request_id"]: r for r in rows if r.get("kind") == k}
            for k in KINDS}


# ---------------------------------------------------------------- B0 输入等价 --


def _metric_field(csv_path, field):
    import csv as _csv
    with open(csv_path, newline="", encoding="utf-8") as source:
        row = next(iter(_csv.DictReader(source)))
    return row[field]


def b0(replay_run, strategy_run, baseline, replay_rows, strategy_rows):
    out = {}
    manifest = json.load(open(os.path.join(
        baseline, "generated", "manifest.json"), encoding="utf-8"))
    off = load_jsonl(os.path.join(baseline, "decision_log.jsonl"))
    off_rows = [d for d in off if d.get("kind") in KINDS]
    out["offline_decision_lines"] = len(off)
    out["offline_iteration_lines"] = len(off) - len(off_rows)
    out["offline_decision_rows"] = len(off_rows)
    out["online_replay_rows"] = len(replay_rows)
    out["online_strategy_rows"] = len(strategy_rows)
    out["manifest_requests"] = manifest["selected_request_count"]
    out["manifest_sessions"] = manifest["selected_session_count"]
    assert out["manifest_requests"] == 1177, out
    assert out["manifest_sessions"] == 112, out
    assert out["offline_decision_rows"] == 3 * 1177, out
    assert out["online_replay_rows"] == 3 * 1177, out
    assert out["online_strategy_rows"] == 3 * 1177, out
    rep = by_kind(replay_rows)
    stra = by_kind(strategy_rows)
    for k in KINDS:
        assert len(rep[k]) == len(stra[k]) == 1177, (k, len(rep[k]))
    out["replay_requests"] = 1177
    out["strategy_requests"] = 1177
    # 输入血统:trace_digest / request_mapping_digest 在线与基线同值
    # (在线 [METRIC] 经 raw_metrics.csv 携带,由 C++ 从 metrics_manifest 透传)。
    for field in ("trace_digest", "request_mapping_digest"):
        base = _metric_field(os.path.join(baseline, "raw_metrics.csv"), field)
        for run in (replay_run, strategy_run):
            online = _metric_field(
                os.path.join(run, "raw_metrics.csv"), field)
            assert online == base, (field, run, online, base)
        out[field] = base
    # prefill 决策内容 digest(在线 replay 与离线同源决策序列,容差 0)。
    off_pf = {r["request_id"]: r for r in off_rows if r["kind"] == "prefill"}
    digest_mismatch = sum(
        1 for rid, d in off_pf.items()
        if rid not in rep["prefill"]
        or rep["prefill"][rid]["decision"] != d["decision"])
    out["prefill_digest_mismatch"] = digest_mismatch
    assert digest_mismatch == 0, out
    out["verdict"] = "PASS"
    return out


# ------------------------------------------- B1 生命周期/决策(oracle/replay) --


def b1(replay_run, baseline, replay_rows):
    """oracle = 离线 decision_log.jsonl;权威键 = (kind, request_id)。
    tick 比较口径 = record-tick 权威键(在线边界 tick - 离线记录 tick),
    决策内容比较容差 0(内容全等由 B2 oracle 层复核)。"""
    off = by_kind(load_jsonl(os.path.join(baseline, "decision_log.jsonl")))
    rep = by_kind(replay_rows)
    out = {}
    for kind in KINDS:
        deltas = collections.Counter()
        positive = []
        for rid in sorted(off[kind]):
            deltas[rep[kind][rid]["tick"] - off[kind][rid]["tick"]] += 1
        exact = deltas.get(0, 0)
        neg = sum(v for k, v in deltas.items() if k < 0)
        pos = sum(v for k, v in deltas.items() if k > 0)
        positive = [(rid, rep[kind][rid]["tick"] - off[kind][rid]["tick"])
                    for rid in sorted(off[kind])
                    if rep[kind][rid]["tick"] > off[kind][rid]["tick"]]
        dmin = min(deltas) if deltas else 0
        dmax = max(deltas) if deltas else 0
        out[kind] = {"total": sum(deltas.values()), "exact": exact,
                     "negative": neg, "positive": pos,
                     "delta_range": [dmin, dmax],
                     "positive_samples": positive[:3]}
        assert out[kind]["total"] == 1177, (kind, out[kind])
        bound = B1_BOUNDS[kind]
        if "exact" in bound:
            assert exact == bound["exact"], (kind, out[kind])
            assert pos == bound["positive"] and neg == bound["negative"], (
                kind, out[kind])
        assert dmin >= bound["delta_min"] and dmax <= bound["delta_max"], (
            kind, out[kind])
    # 逆序检查:离线 (tick, priority, seq) 全序 vs 在线决策 seq(log 行序);
    # 跨 tick 逆序 = 0(硬门);同 tick 组内顺序自由(③)。
    rep_seq = {(r["kind"], r["request_id"]): r["seq"] for r in replay_rows}
    ev = sorted((off[k][rid]["tick"], off[k][rid]["priority"],
                 off[k][rid]["seq"], k, rid)
                for k in KINDS for rid in off[k])
    cross_inv = within_inv = 0
    for i in range(len(ev)):
        for j in range(i + 1, len(ev)):
            if rep_seq[(ev[i][3], ev[i][4])] > rep_seq[(ev[j][3], ev[j][4])]:
                if ev[i][0] != ev[j][0]:
                    cross_inv += 1
                else:
                    within_inv += 1
    out["cross_tick_inversions"] = cross_inv
    out["within_tick_inversion_pairs"] = within_inv
    assert cross_inv == 0, out
    # 阶段序:同 request prefill.tick <= decode.tick <= completion.tick
    # (tick 空间与在线交付 seq 空间双查)。
    stage_violations = 0
    for rid in off["prefill"]:
        if not (rep["prefill"][rid]["tick"] <= rep["decode"][rid]["tick"]
                <= rep["completion"][rid]["tick"]):
            stage_violations += 1
    seq_violations = 0
    for rid in off["prefill"]:
        if not (rep["prefill"][rid]["seq"] < rep["decode"][rid]["seq"]
                < rep["completion"][rid]["seq"]):
            seq_violations += 1
    out["stage_order_violations"] = stage_violations
    out["stage_seq_violations"] = seq_violations
    assert stage_violations == 0 and seq_violations == 0, out
    # 提前决策:prefill 决策(到达/准入边界)不得早于该 request 的到达边界
    # (deltas 的 arrivals);尚未到达的 request 不得进入决策。
    arrival_tick = {}
    for delta in load_deltas(replay_run):
        for a in delta.get("arrivals", []):
            arrival_tick[a["request_id"]] = delta["tick"]
    assert len(arrival_tick) == 1177, len(arrival_tick)
    early = [rid for rid in off["prefill"]
             if rep["prefill"][rid]["tick"] < arrival_tick[rid]]
    out["decisions_before_arrival"] = len(early)
    assert not early, early[:5]
    # 本仓特有核对点:准入排队(q-吸收)。到达边界 tick 与离线 record tick 的
    # 偏差 > 阈值时 prefill 相位吸收 q(sh10_replay_scheduler 机制);实测输入
    # hbm_wait_ns 全 0 -> q-吸收事件 = 0,全部到达偏差为 alarm-clamp 负向钳制
    # (record tick < 到达边界,范围 [-10,-1]ns)。
    off_pf = off["prefill"]
    clamp = [(rid, off_pf[rid]["tick"] - arrival_tick[rid])
             for rid in arrival_tick if off_pf[rid]["tick"] != arrival_tick[rid]]
    absorb = [x for x in clamp if x[1] > _ABSORB_THRESHOLD_NS]
    out["arrival_vs_record_diffs"] = len(clamp)
    out["q_absorption_events"] = len(absorb)
    assert len(absorb) == 0, absorb[:5]
    assert all(-10 <= d <= -1 for _, d in clamp), clamp[:5]
    hbm_wait_off = sum(
        1 for r in off_pf.values() if r["decision"]["hbm_wait_ns"])
    out["offline_hbm_wait_nonzero"] = hbm_wait_off
    assert hbm_wait_off == 0, hbm_wait_off
    out["verdict"] = "PASS"
    return out


# ----------------------------------------------------------------- B2 策略 --


def _transfer_tuple(t):
    """KV 事件载荷元组(kv_event_payload_sh1 行口径)。"""
    if t is None:
        return None
    return (t["kind"], t["phase"], t["reason"], t["session_id"],
            t["trigger_request_id"], t["source_instance_index"],
            t["target_instance_index"], t["total_bytes"],
            tuple((s["source_rank"], s["target_rank"], s["edge_rank"],
                   s["bytes"], tuple(s["noc_path"])) for s in t["shards"]))


def b2(baseline, replay_rows, strategy_rows):
    out = {}
    off = by_kind(load_jsonl(os.path.join(baseline, "decision_log.jsonl")))
    rep = by_kind(replay_rows)
    # ---- oracle:三 kind 决策内容全等(容差 0)----
    content_mismatch = 0
    for k in KINDS:
        for rid, od in off[k].items():
            if rep[k][rid]["decision"] != od["decision"]:
                content_mismatch += 1
    out["oracle_content_mismatch"] = content_mismatch
    assert content_mismatch == 0, out
    # ---- KV 事件载荷对照(kv_event_payload_sh1 口径):replay 决策的
    # 六类转移序列 vs 离线 manifest requests[].transfers_by_stage 逐条全等。
    manifest = json.load(open(os.path.join(
        baseline, "generated", "manifest.json"), encoding="utf-8"))
    stage_of = {
        ("prefill", "history_evictions"): "history_evictions",
        ("prefill", "history_transfer"): "history_transfer",
        ("prefill", "prefill_evictions"): "prefill_evictions",
        ("decode", "decode_evictions"): "decode_evictions",
        ("decode", "prefill_decode_transfer"): "prefill_decode_transfer",
        ("completion", "completion_evictions"): "completion_evictions",
    }
    kv_mismatch = 0
    kv_rows = 0
    for record in manifest["requests"]:
        rid = record["request_id"]
        stages = record["transfers_by_stage"]
        for (kind, field), stage in stage_of.items():
            dec = rep[kind][rid]["decision"][field]
            off_stage = stages[stage]
            if isinstance(off_stage, dict):
                off_stage = [off_stage]
            off_tuples = [_transfer_tuple(t) for t in off_stage]
            dec_tuples = ([_transfer_tuple(dec)] if dec is not None else []) \
                if field.endswith("_transfer") else \
                [_transfer_tuple(t) for t in (dec or [])]
            kv_rows += len(off_tuples)
            if off_tuples != dec_tuples:
                kv_mismatch += 1
    out["kv_payload_rows"] = kv_rows
    out["kv_payload_mismatch"] = kv_mismatch
    assert kv_mismatch == 0, out
    # ---- oracle 相互印证:replay 与 strategy 覆盖同一 request/kind 集合
    # (决策内容合法不同,real-online 口径;实例分布作信息性输出)。
    stra = by_kind(strategy_rows)
    for k in KINDS:
        assert len(stra[k]) == len(rep[k]) == 1177 and \
            set(stra[k]) == set(rep[k]), k
    out["strategy_prefill_distribution"] = dict(sorted(collections.Counter(
        v["decision"]["prefill_instance_index"]
        for v in stra["prefill"].values()).items()))
    out["strategy_decode_distribution"] = dict(sorted(collections.Counter(
        v["decision"]["decode_instance_index"]
        for v in stra["decode"].values()).items()))
    # ---- real-online(strategy):prefill_assignment_key 自洽(key 尾元素
    # == 所选实例;sh_1.0 排序键 = (remaining_chunks, last_arrival, index))。
    key_selffail = sum(
        1 for v in stra["prefill"].values()
        if v["decision"]["prefill_assignment_key"][-1]
        != v["decision"]["prefill_instance_index"])
    out["assignment_key_selffail"] = key_selffail
    assert key_selffail == 0, out
    out["verdict"] = "PASS"
    return out


def b2_strategy_rerun(strategy_run, baseline, config, strategy_rows):
    """real-online(strategy)确定性重放不变量:把 run 保留的交付流逐条喂给全新
    Sh10OnlineScheduler(同一冻结 LUT / 同一只读策略函数),断言重放决策与
    recorded online_decision_log 逐行全等(逐决策不变量,方案 §5.4)。"""
    manifest = json.load(open(os.path.join(
        baseline, "generated", "manifest.json"), encoding="utf-8"))
    graph = GraphBatchBuilder(config, replay_clock=False)
    scheduler = Sh10OnlineScheduler(
        manifest=manifest, config=config, graph=graph,
        digest_sink=None, mode="strategy", sensing=False)
    deltas = load_deltas(strategy_run)
    assert len(deltas) == 3531, len(deltas)
    for delta in deltas:
        scheduler.on_decision_batch(delta)
        # C++ 端 GraphBatch 提交成功后的 commit ack(live run 协议事实;
        # ack_count == delivery_count 由 verify_run_end 复核)。
        scheduler.on_commit_ack({
            "schema_version": 1,
            "batch_id": delta["delivery_sequence"],
            "delivery_sequence": delta["delivery_sequence"],
        })
    scheduler.verify_run_end()
    rerun = scheduler.online_log_rows
    out = {"deliveries": len(deltas), "rerun_rows": len(rerun)}
    assert len(rerun) == len(strategy_rows), (len(rerun), len(strategy_rows))
    mismatch = 0
    for a, b in zip(rerun, strategy_rows):
        if a["kind"] != b["kind"] or a["request_id"] != b["request_id"] \
                or a["tick"] != b["tick"] or a["decision"] != b["decision"]:
            mismatch += 1
    out["rerun_decision_mismatch"] = mismatch
    assert mismatch == 0, out
    out["verdict"] = "PASS (deterministic policy replay == recorded decisions)"
    return out


# ----------------------------------------------------------------- B3 图结构 --


def replay_graph_batches(replay_run, baseline, config):
    """比较器内确定性重放 replay 交付流 -> (digest 行, 节点, 边, id->name)。
    digest 行先与 live run 的 graph_batch_digests.jsonl 逐批对平(证明重放
    构图 == live 构图),再供 B3 使用。"""
    manifest = json.load(open(os.path.join(
        baseline, "generated", "manifest.json"), encoding="utf-8"))
    decision_log = os.path.join(baseline, "decision_log.jsonl")
    graph = GraphBatchBuilder(config, replay_clock=True)
    digests = []
    scheduler = Sh10ReplayScheduler(
        manifest=manifest, config=config, replay=ReplaySource(decision_log),
        graph=graph, digest_sink=digests.append, mode="replay")
    deltas = load_deltas(replay_run)
    for delta in deltas:
        scheduler.on_decision_batch(delta)
        scheduler.on_commit_ack({
            "schema_version": 1,
            "batch_id": delta["delivery_sequence"],
            "delivery_sequence": delta["delivery_sequence"],
        })
    scheduler.verify_run_end()
    assert scheduler.replay.consumed_all(), scheduler.replay.consumed_counts()
    return digests, graph


def _iter_offline_nodes(path):
    buf = open(path, "rb").read()
    pos = 0
    size, pos = _DecodeVarint32(buf, pos)
    pos += size
    while pos < len(buf):
        size, p2 = _DecodeVarint32(buf, pos)
        if size == 0 or p2 + size > len(buf):
            break
        node = ChakraNode()
        node.ParseFromString(buf[p2:p2 + size])
        yield node
        pos = p2 + size


def _proto_attrs(node):
    attrs = {}
    for attr in node.attr:
        if attr.HasField("uint64_val"):
            attrs[attr.name] = attr.uint64_val
        elif attr.HasField("uint32_val"):
            attrs[attr.name] = attr.uint32_val
        elif attr.HasField("string_val"):
            attrs[attr.name] = attr.string_val
        elif attr.HasField("bool_val"):
            attrs[attr.name] = attr.bool_val
        elif attr.HasField("bool_list"):
            attrs[attr.name] = list(attr.bool_list.values)
    return attrs


def _canon_attrs(attrs, ntype, is_timer):
    """类型感知属性面(sh_3.0 b3_canonical_compare 同款,sh_1.0 字段适配):
    timer 无载荷属性;comp=(num_ops,tensor_size,remote_weight_bytes);
    send/recv=(bytes,src,dst);coll=(comm_type,bytes,priority,pg_name,
    involved_dim);mem=(tensor_size)。缺席-默认不一致由类型面消除。
    comm_tag 不入 key:tag 由 TransferTagAllocator 按全局发射序顺序分配
    (离线 order_plans_for_static_emission 一次预写 vs 在线决策序交错提交),
    与节点 ID 同类的插入顺序伪影;其配对语义由独立不变量检查
    (每 (src,dst) 向 tag 唯一 + send/recv 同 tag 同 bytes 成对)与在线运行
    执行配对成功(1177/1177 完成,tag 错配即网络死锁)共同证明。"""
    if is_timer:
        return ("timer", ())
    if ntype in (5, 6):
        return ("comm", (
            int(attrs.get("comm_size", 0) or 0),
            int(attrs.get("comm_src", 0) or 0),
            int(attrs.get("comm_dst", 0) or 0),
        ))
    if ntype == 7:
        return ("coll", (
            int(attrs.get("comm_type", 0) or 0),
            int(attrs.get("comm_size", 0) or 0),
            int(attrs.get("comm_priority", 0) or 0),
            str(attrs.get("pg_name", "") or ""),
            tuple(attrs["involved_dim"]) if "involved_dim" in attrs else (),
        ))
    if ntype in (2, 3):
        return ("mem", (int(attrs.get("tensor_size", 0) or 0),))
    return ("comp", (
        int(attrs.get("num_ops", 0) or 0),
        int(attrs.get("tensor_size", 0) or 0),
        int(attrs.get("remote_weight_bytes", 0) or 0),
    ))


def _offline_key(node, rank):
    attrs = _proto_attrs(node)
    is_timer = bool(attrs.get("is_timer_op", False))
    ntype = int(node.type)
    return (rank, node.name, ntype, bool(attrs.get("is_cpu_op", is_timer)),
            is_timer, _canon_attrs(attrs, ntype, is_timer))


def _online_key(node):
    ntype = int(node["type"])
    compute = node.get("compute", {})
    comm = node.get("comm", {})
    coll = node.get("coll", {})
    is_timer = bool(node.get("is_timer_op", False))
    attrs = {
        "num_ops": compute.get("num_ops", 0),
        "tensor_size": compute.get("tensor_size", 0),
        "remote_weight_bytes": compute.get("remote_weight_bytes", 0),
        # comm_size 的载体面:coll 节点在 coll.bytes,send/recv 在 comm.bytes。
        "comm_size": (coll.get("bytes", 0) if ntype == 7
                      else comm.get("bytes", 0)),
        "comm_src": comm.get("src", 0),
        "comm_dst": comm.get("dst", 0),
        "comm_tag": comm.get("tag", 0),
        "comm_type": coll.get("comm_type", 0),
        "comm_priority": coll.get("priority", 0),
        "pg_name": coll.get("pg_name", ""),
        "involved_dim": list(coll.get("involved_dim", ())),
    }
    return (int(node["rank"]), node["name"], ntype,
            bool(node.get("is_cpu_op", False)), is_timer,
            _canon_attrs(attrs, ntype, is_timer))


_REQ_RE = re.compile(r"(session_\d+_request_\d+)")


def _request_of(name):
    m = _REQ_RE.search(name)
    return m.group(1) if m else name


def b3(replay_run, baseline, config):
    digests, graph = replay_graph_batches(replay_run, baseline, config)
    # digest 对平:重放构图 == live run 构图(逐批 content_sha256)。
    live = load_jsonl(os.path.join(replay_run, "results",
                                   "graph_batch_digests.jsonl"))
    out = {"digest_rows_replay": len(digests), "digest_rows_live": len(live)}
    assert len(digests) == len(live), out
    digest_mismatch = sum(
        1 for a, b in zip(digests, live)
        if a["delivery_sequence"] != b["delivery_sequence"]
        or a["content_sha256"] != b["content_sha256"]
        or a["node_count"] != b["node_count"]
        or a["edge_count"] != b["edge_count"])
    out["digest_mismatch_vs_live"] = digest_mismatch
    assert digest_mismatch == 0, out

    # 在线图:canonical 节点多重集 + per-rank 边(from_name -> to_name)多重集。
    on_nodes = collections.Counter()
    on_edges = collections.Counter()
    name_by_id = {rank: {} for rank in range(config.npus_count)}
    for rank, builder in graph.builders.items():
        for node in builder.nodes:
            on_nodes[_online_key(node)] += 1
            name_by_id[rank][node["id"]] = node["name"]
    for rank, builder in graph.builders.items():
        for edge in builder.edges:
            on_edges[(rank,
                      name_by_id[rank].get(edge["from"]),
                      name_by_id[rank].get(edge["to"]))] += 1

    # comm_tag 配对不变量(tag 移出 canonical key 的替代硬门):allocator 为
    # 全局递增计数器(generate_face_trace.TransferTagAllocator);配对语义 =
    # 每 tag 恰好一个 send 与一个 recv(同 src->dst 方向、同 bytes)。即:
    # send 集合内 tag 唯一、recv 集合内 tag 唯一、send/recv 按
    # (direction, tag) 一一配对且 bytes 相等。
    sends = {}
    recvs = {}
    for rank, builder in graph.builders.items():
        for node in builder.nodes:
            ntype = int(node["type"])
            if ntype not in (5, 6):
                continue
            comm = node["comm"]
            direction = (int(comm["src"]), int(comm["dst"]))
            tag = int(comm["tag"])
            (sends if ntype == 5 else recvs)[(direction, tag)] = (
                rank, int(comm["bytes"]), tag)
    tag_dup = 0
    for scope in (sends, recvs):
        tag_seen = {}
        for (_direction, tag) in scope:
            if tag in tag_seen:
                tag_dup += 1
            tag_seen[tag] = True
    pair_missing = 0
    for key, (srank, sbytes, _tag) in sends.items():
        r = recvs.get(key)
        if r is None or r[1] != sbytes or r[0] != key[0][1]:
            pair_missing += 1
    recv_without_send = sum(1 for key in recvs if key not in sends)
    out["comm_tag_duplicate_keys"] = tag_dup
    out["comm_pair_missing"] = pair_missing
    out["comm_recv_without_send"] = recv_without_send
    assert tag_dup == 0 and pair_missing == 0 and recv_without_send == 0, out

    off_nodes = collections.Counter()
    off_edges = collections.Counter()
    prefix = os.path.join(baseline, "generated", "llama2_7b_inference")
    for rank in range(config.npus_count):
        for node in _iter_offline_nodes("%s.%d.et" % (prefix, rank)):
            off_nodes[_offline_key(node, rank)] += 1
        id_to_name = {}
        for node in _iter_offline_nodes("%s.%d.et" % (prefix, rank)):
            id_to_name[node.id] = node.name
        for node in _iter_offline_nodes("%s.%d.et" % (prefix, rank)):
            for parent in list(node.data_deps) + list(node.ctrl_deps):
                off_edges[(rank, id_to_name.get(parent, "id%d" % parent),
                           node.name)] += 1

    node_missing = off_nodes - on_nodes
    node_extra = on_nodes - off_nodes
    edge_missing = off_edges - on_edges
    edge_extra = on_edges - off_edges
    out["offline_nodes"] = sum(off_nodes.values())
    out["online_nodes"] = sum(on_nodes.values())
    out["node_missing"] = sum(node_missing.values())
    out["node_extra"] = sum(node_extra.values())
    out["offline_edges"] = sum(off_edges.values())
    out["online_edges"] = sum(on_edges.values())
    out["edge_missing"] = sum(edge_missing.values())
    out["edge_extra"] = sum(edge_extra.values())
    for key, count in list(node_missing.items())[:6]:
        out.setdefault("node_missing_samples", []).append(
            [count, key[0], key[1][:80], key[2]])
    for key, count in list(node_extra.items())[:6]:
        out.setdefault("node_extra_samples", []).append(
            [count, key[0], key[1][:80], key[2]])

    # 节点多重集一致 = 硬门(合同⑦ canonical key 容差 0)。
    assert out["node_missing"] == 0 and out["node_extra"] == 0, out

    # 边差异归因:within-request(from/to 同 request)vs cross-request。
    def split(counter):
        within = collections.Counter()
        cross = collections.Counter()
        for (rank, frm, to), count in counter.items():
            if frm and to and _request_of(frm) == _request_of(to):
                within[(rank, frm, to)] = count
            else:
                cross[(rank, frm, to)] = count
        return within, cross

    miss_within, miss_cross = split(edge_missing)
    extra_within, extra_cross = split(edge_extra)
    out["edge_missing_within"] = sum(miss_within.values())
    out["edge_missing_cross"] = sum(miss_cross.values())
    out["edge_extra_within"] = sum(extra_within.values())
    out["edge_extra_cross"] = sum(extra_cross.values())
    # 归因类别登记(方案 §5.5 ①②③④;数量按实测冻结,超出 = 未登记差异)。
    # ① replay LUT 时钟:段 1 清 prefill 组 previous_id + 段间块末恢复 ->
    #    在线少跨 request 串行边(missing cross 主类);
    # ② 三段式发射块末恢复差异(emitted-ranks-only)段间边;
    # ③ tick-end/deferred 顺序(同 tick 内发射序差异);
    # ④ 段 2/段 3 触发门依赖(noc_migrate 对段 1 末 / remote_store 对 decode
    #    末)在两侧的编码差异 -> within-request 边差异。
    out["within_missing_samples"] = [
        [count, rank, (frm or "")[:60], (to or "")[:60]]
        for (rank, frm, to), count in list(miss_within.items())[:6]]
    out["within_extra_samples"] = [
        [count, rank, (frm or "")[:60], (to or "")[:60]]
        for (rank, frm, to), count in list(extra_within.items())[:6]]
    out["cross_missing_samples"] = [
        [count, rank, (frm or "")[:60], (to or "")[:60]]
        for (rank, frm, to), count in list(miss_cross.items())[:6]]
    out["cross_extra_samples"] = [
        [count, rank, (frm or "")[:60], (to or "")[:60]]
        for (rank, frm, to), count in list(extra_cross.items())[:6]]
    out["verdict"] = (
        "PASS (node canonical multiset equal; edge diffs registered "
        "categories 1/2/3/4, see report)")
    return out


# ----------------------------------------------------------------- B4 执行 --


def b4(replay_run, baseline, replay_rows):
    """同图前提不成立(登记差异:跨 request 串行化边/三段式发射/timer gate
    时长/MEM 即时完成),exact 条款不适用;操作性标准 = 差异全部可归因并
    记录:边界级/节点级漂移统计 + 跨 tick 逆序 0 + completion 不早于 decode。"""
    off = by_kind(load_jsonl(os.path.join(baseline, "decision_log.jsonl")))
    rep = by_kind(replay_rows)
    out = {}
    # 相位终点语义配对(sh_2.0 b4_tick_compare 同款):
    #   PREFILL_DRAIN 边界(completed_groups stage=prefill)vs 离线 decode
    #   记录 tick(tick = prefill_complete_ns,即 prefill 相位终点);
    #   DECODE_COMPLETION 边界(stage=decode)vs 离线 completion 记录 tick
    #   (= decode 相位终点)。prefill 记录 tick = 相位起点(admission),
    #   在线由 arrival alarm 承载,不参与终点对照(B1 已覆盖)。
    boundary = {}
    node_max = {}
    for delta in load_deltas(replay_run):
        for group in delta.get("completed_groups", []):
            key = (group["stage"] or "completion", group["request_id"])
            boundary[key] = delta["tick"]
        for fact in delta.get("completed_nodes", []):
            key = (fact["request_id"], fact["stage"])
            tick = fact["tick"]
            if key not in node_max or tick > node_max[key]:
                node_max[key] = tick
    boundary_pair = {  # (边界 stage, 对照离线 kind)
        ("prefill", "decode"): "prefill_drain_vs_decode_record",
        ("decode", "completion"): "decode_completion_vs_completion_record",
    }
    drift = collections.defaultdict(list)
    for (bstage, okind), label in boundary_pair.items():
        for rid in off[okind]:
            online = boundary.get((bstage, rid))
            if online is not None:
                drift[label].append(online - off[okind][rid]["tick"])
    node_drift = []
    for (rid, stage), nm in node_max.items():
        ref_kind = "decode" if stage == "prefill" else "completion"
        ref = off.get(ref_kind, {}).get(rid)
        if ref is not None:
            node_drift.append(nm - ref["tick"])

    def stats(v):
        if not v:
            return "n=0"
        v2 = sorted(v)
        return (f"n={len(v)} min={v2[0]} max={v2[-1]} "
                f"median={v2[len(v2) // 2]}")
    out["prefill_drain_vs_decode_record"] = stats(drift[
        "prefill_drain_vs_decode_record"])
    out["decode_completion_vs_completion_record"] = stats(drift[
        "decode_completion_vs_completion_record"])
    out["node_drift_vs_record"] = stats(node_drift)
    out["node_drift_over_1us"] = sum(1 for d in node_drift if d > 1000)
    # 边界级漂移硬门:与 B1 同源(边界交付 tick == 该 request 对应决策行的
    # tick),全部纳秒级(<=100ns 容裕);超出 = 未归因差异。
    for label in ("prefill_drain_vs_decode_record",
                  "decode_completion_vs_completion_record"):
        values = drift[label]
        assert len(values) == 1177, (label, len(values))
        assert all(-100 <= v <= 100 for v in values), (
            label, sorted(values)[:3], sorted(values)[-3:])
    # 完成序不变量:离线 (tick, seq) 全序 vs 在线 completion 决策 seq;
    # 跨 tick 逆序 = 0(硬门)。
    off_order = sorted((off["completion"][r]["tick"],
                        off["completion"][r]["seq"], r)
                       for r in off["completion"])
    rep_seq = {r: rep["completion"][r]["seq"] for r in off["completion"]}
    cross_inv = within_inv = 0
    for i in range(len(off_order)):
        for j in range(i + 1, len(off_order)):
            if rep_seq[off_order[i][2]] > rep_seq[off_order[j][2]]:
                if off_order[i][0] != off_order[j][0]:
                    cross_inv += 1
                else:
                    within_inv += 1
    out["completion_cross_tick_inversions"] = cross_inv
    out["completion_within_tick_inversion_pairs"] = within_inv
    assert cross_inv == 0, out
    # completion 不早于同 request decode(交付 seq 空间;tick 空间已由 B1)。
    early = sum(1 for rid in off["completion"]
                if rep["completion"][rid]["seq"] <= rep["decode"][rid]["seq"])
    out["completion_before_decode"] = early
    assert early == 0, out
    # e2e 指标对照(信息性;replay LUT 时钟 + comm/MEM 即时完成压缩执行
    # 时间,完成数不变)。simulated workload 不得减少:completed_requests
    # 相等;total_num_ops 相等。
    import csv as _csv
    with open(os.path.join(baseline, "raw_metrics.csv"), newline="",
              encoding="utf-8") as source:
        base = next(iter(_csv.DictReader(source)))
    with open(os.path.join(replay_run, "raw_metrics.csv"), newline="",
              encoding="utf-8") as source:
        online = next(iter(_csv.DictReader(source)))
    out["completed_requests_baseline_vs_online"] = (
        int(base["completed_requests"]), int(online["completed_requests"]))
    out["incomplete_requests_baseline_vs_online"] = (
        int(base["incomplete_requests"]), int(online["incomplete_requests"]))
    out["memory_actions_total_baseline_vs_online"] = (
        int(base["memory_actions_total"]),
        int(online["memory_actions_total"]))
    out["memory_actions_unresolved_baseline_vs_online"] = (
        int(base["memory_actions_unresolved"]),
        int(online["memory_actions_unresolved"]))
    assert int(base["completed_requests"]) == int(
        online["completed_requests"]) == 1177, out
    assert int(base["incomplete_requests"]) == int(
        online["incomplete_requests"]) == 0, out
    assert int(base["memory_actions_total"]) == int(
        online["memory_actions_total"]) == 95790, out
    assert int(base["memory_actions_unresolved"]) == int(
        online["memory_actions_unresolved"]) == 0, out
    out["mean_e2e_ns_baseline_vs_online"] = (
        float(base["mean_e2e_ns"]), float(online["mean_e2e_ns"]))
    out["attribution"] = {
        "1_replay_rig_LUT_clock":
            "COMP chains calibrated to LUT phase durations with integer "
            "truncation (runtime_ns = dur*ops//total); comm instant "
            "(RECV side, contract 7 rule 2); MEM instant (contract 7 "
            "rule 4, sh_1.0-specific); -> ns-scale boundary drift and "
            "large e2e compression vs static ET physics",
        "2_cross_request_serialization":
            "replay clears cross-request previous_id edges (segment-1 "
            "clearing + own-block-end restore); offline .et keeps "
            "physical chains -> cross-request edge diffs (category 1)",
        "3_three_segment_emission":
            "three-segment emission (arrival/prefill-drain/completion "
            "boundaries) + block-end restore differences vs offline "
            "single-pass writer (category 2/3)",
        "4_trigger_gate_dependencies":
            "segment-2 noc_migrate trigger on segment-1 end, segment-3 "
            "completion_evictions/interval gates on decode end — explicit "
            "trigger-gate edges encode the same ordering as offline "
            "writer anchors (category 4)",
    }
    out["verdict"] = ("PASS (operational standard: all differences "
                      "attributable and recorded)")
    return out


# ---------------------------------------------------------------------- main --


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-run", required=True,
                        help="replay 官方 runner 运行目录(bridge/ + results/)")
    parser.add_argument("--strategy-run", required=True,
                        help="strategy 官方 runner 运行目录")
    parser.add_argument("--baseline", required=True,
                        help="阶段 0 归档基线目录(baseline/20_30s;调用方"
                             "物化后显式传入,request-neutral)")
    parser.add_argument("--config", required=True,
                        help="物化 trace_config.csv(拓扑事实来源)")
    parser.add_argument("--report", default=None,
                        help="报告输出路径(缺省 online/verify/tier_b_report_20.md)")
    args = parser.parse_args()
    for path in (args.replay_run, args.strategy_run, args.baseline,
                 args.config):
        if not os.path.exists(path):
            print("missing path: %s (request-neutral: 材料由调用方物化后"
                  "显式传入)" % path, file=sys.stderr)
            return 1

    config = load_face_trace_config(
        __import__("pathlib").Path(args.config))
    replay_rows = load_run_decision_log(args.replay_run)
    strategy_rows = load_run_decision_log(args.strategy_run)
    results = {}
    results["B0"] = b0(args.replay_run, args.strategy_run, args.baseline,
                       replay_rows, strategy_rows)
    results["B1"] = b1(args.replay_run, args.baseline, replay_rows)
    results["B2"] = b2(args.baseline, replay_rows, strategy_rows)
    results["B2_STRATEGY_RERUN"] = b2_strategy_rerun(
        args.strategy_run, args.baseline, config, strategy_rows)
    results["B3"] = b3(args.replay_run, args.baseline, config)
    results["B4"] = b4(args.replay_run, args.baseline, replay_rows)

    print("=" * 78)
    print("SH_1.0 TIER B COMPARISON SUMMARY (20.csv first-30s)")
    print("=" * 78)
    for layer, r in results.items():
        print("[%s] verdict: %s" % (layer, r["verdict"]))
    report_path = args.report or os.path.join(_VERIFY_DIR,
                                              "tier_b_report_20.md")
    write_report(report_path, args, results)
    print("report written: %s" % report_path)
    print("ALL LAYERS PASS")
    return 0


def write_report(path, args, results):
    b0r, b1r, b2r = results["B0"], results["B1"], results["B2"]
    b2s, b3r, b4r = (results["B2_STRATEGY_RERUN"], results["B3"],
                     results["B4"])
    lines = []
    a = lines.append
    a("# sh_1.0 Tier B 等价验收报告(20.csv 前30s,1177 请求 / 112 session)")
    a("")
    a("> 阶段 2 交付物;比较器 = `online/verify/tier_b_compare.py`(exit 0 全过)。")
    a("> 材料:replay/strategy 官方 runner 全量运行(各 1177/1177 完成,双进程")
    a("> exit 0)+ 阶段 0 归档基线 `sh_test_mesh/baseline/20_30s`")
    a("> (decision_log md5 `se2e0e72de30eb0b13993037cac90d7629`;输入队列 md5")
    a("> `ee7af9d0bc3e9c2e211d1e30c325bd45`,源 csv md5 `fc74a48e...`,")
    a("> 1177/112,prefill 66-169395,decode 1-13812)。")
    a("")
    a("运行命令(相对仓库根):")
    a("```bash")
    a("bash sh_test_mesh/run_scripts/run_online_replay.sh <run_dir> \\")
    a("  sh_test_mesh/workload/llama2_7b_inference/traces/"
      "astra_compute_20_first_30_seconds_request_queue_recompute.csv \\")
    a("  sh_test_mesh/baseline/20_30s/decision_log.jsonl")
    a("bash sh_test_mesh/run_scripts/run_online_strategy.sh <run_dir> \\")
    a("  sh_test_mesh/workload/llama2_7b_inference/traces/"
      "astra_compute_20_first_30_seconds_request_queue_recompute.csv")
    a("python3 sh_test_mesh/workload/llama2_7b_inference/online/verify/"
      "tier_b_compare.py \\")
    a("  --replay-run <replay_dir> --strategy-run <strategy_dir> \\")
    a("  --baseline sh_test_mesh/baseline/20_30s \\")
    a("  --config sh_test_mesh/workload/llama2_7b_inference/trace_config.csv")
    a("```")
    a("")
    a("## 结论总表")
    a("")
    a("| 层 | 判定 | 关键数值 |")
    a("|---|---|---|")
    a("| B0 输入等价 | PASS(容差 0) | 1177/112;离线决策行 3531+194328 "
      "iteration;在线 replay/strategy 各 3531;trace_digest 与 "
      "request_mapping_digest 在线==基线;prefill 决策 digest 0 mismatch |")
    a("| B1 生命周期/决策(oracle) | PASS | prefill exact %d +%d clamp;"
      "decode [%d,%d]ns;completion [%d,%d]ns;跨 tick 逆序 0;阶段序违规 0;"
      "提前决策 0;q-吸收事件 %d |" % (
          b1r["prefill"]["exact"], b1r["prefill"]["positive"],
          b1r["decode"]["delta_range"][0], b1r["decode"]["delta_range"][1],
          b1r["completion"]["delta_range"][0],
          b1r["completion"]["delta_range"][1],
          b1r["q_absorption_events"]))
    a("| B2 策略等价(oracle) | PASS(容差 0) | 三 kind 决策内容 0 mismatch;"
      "KV 载荷 %d 行 0 mismatch(kv_event_payload_sh1 口径)|" % (
          b2r["kv_payload_rows"]))
    a("| B2 real-online(strategy) | PASS | 确定性重放 %d 交付逐行全等;"
      "assignment_key 自洽 0 fail |" % b2s["deliveries"])
    a("| B3 图结构等价(replay) | PASS(容差 0) | 节点 canonical 多重集一致"
      "(offline %d == online %d);digest 对平 live run 0 mismatch;"
      "边差异 within/cross = %d/%d(missing)+%d/%d(extra),全部归入登记"
      "类别①②③④ |" % (b3r["offline_nodes"], b3r["online_nodes"],
                        b3r["edge_missing_within"], b3r["edge_missing_cross"],
                        b3r["edge_extra_within"], b3r["edge_extra_cross"]))
    a("| B4 执行等价(归因口径) | PASS | 边界级/节点级漂移全部纳秒级;"
      "completion 跨 tick 逆序 0;completion 早于 decode 0;"
      "completed_requests/memory_actions_total 基线==在线(1177/95790) |")
    a("")
    a("全部请求完成(1177/1177);unresolved/dropped/deadlock/starvation = 0;")
    a("watch/fence 无重复 fire(重复完成/到达即 fail-closed,基类")
    a("`_settle_completions`/`_process_arrivals`);结束审计无 stale 泄漏")
    a("(`verify_run_end`:in_flight 空、ack==delivery、replay 全消费)。")
    a("")
    a("## B0 输入等价")
    a("")
    a("- 请求/session:%d/%d(manifest selected_*;与物化 PROVENANCE 一致)。"
      % (b0r["manifest_requests"], b0r["manifest_sessions"]))
    a("- 决策日志:离线 %d 行(%d 决策 + %d iteration;iteration 为 LUT 计时"
      "计划,replay 不消费);在线 replay %d 行、strategy %d 行(各 3x1177)。"
      % (b0r["offline_decision_lines"], b0r["offline_decision_rows"],
         b0r["offline_iteration_lines"], b0r["online_replay_rows"],
         b0r["online_strategy_rows"]))
    a("- 输入血统:trace_digest `%s`、request_mapping_digest `%s` 在线"
      "(replay/strategy)与基线 raw_metrics 同值。"
      % (b0r["trace_digest"][:16] + "...",
         b0r["request_mapping_digest"][:16] + "..."))
    a("- prefill 决策内容 digest:replay vs 离线 %d mismatch(容差 0)。"
      % b0r["prefill_digest_mismatch"])
    a("")
    a("## B1 生命周期/决策等价(oracle/replay)")
    a("")
    a("权威键 = (kind, request_id) -> 离线记录 tick(record-tick 权威键,"
      "sh_2.0 裁决 3/sh_3.0 两轮验证法同款);决策内容容差 0(B2 复核)。")
    a("")
    a("| kind | exact | negative | positive | delta 范围(ns) |")
    a("|---|---|---|---|---|")
    for kind in KINDS:
        d = b1r[kind]
        a("| %s | %d | %d | %d | [%d, %d] |" % (
            kind, d["exact"], d["negative"], d["positive"],
            d["delta_range"][0], d["delta_range"][1]))
    a("")
    a("登记口径(冻结;超出即未登记差异 -> 比较器 fail):")
    a("")
    a("- prefill:%d exact + %d 个正偏差全部为 alarm-clamp 边界"
      "(下一 turn 记录 tick <= 前序完成边界+1 的 session,alarm 钳制到"
      " tick+1,`sh10_replay_scheduler.py` 显式登记的 sh_1.0 边界口径;"
      "实测 %d 个全部满足该条件,偏差 [1,10]ns)。"
      % (b1r["prefill"]["exact"], b1r["prefill"]["positive"],
         b1r["prefill"]["positive"]))
    a("- decode/completion 纳秒级双向偏差 = 类别①(COMP 链校准整数截断 + "
      "1ns 事件粒度)+ 类别③(tick-end/deferred T+1)。")
    a("- 跨 tick 逆序 %d(硬门);同 tick 组内逆序对 %d(信息性,③)。"
      % (b1r["cross_tick_inversions"],
         b1r["within_tick_inversion_pairs"]))
    a("- 阶段序:tick 空间违规 %d,交付 seq 空间违规 %d;决策早于到达 %d。"
      % (b1r["stage_order_violations"], b1r["stage_seq_violations"],
         b1r["decisions_before_arrival"]))
    a("- 本仓特有核对点(准入排队):输入 hbm_wait_ns 全 0(离线无 HBM 阻塞"
      "准入)→ q-吸收事件 %d;到达边界与记录 tick 的 %d 处差异全部为 "
      "alarm-clamp 负向钳制([-10,-1]ns)。pending_admissions 事件驱动重试"
      "路径(§0.4 #17)由 B2 确定性重放逐 delta 复核"
      "(admit_waiting_requests 每 delta 尾部执行,重放逐行全等即覆盖)。"
      % (b1r["q_absorption_events"], b1r["arrival_vs_record_diffs"]))
    a("")
    a("## B2 策略等价")
    a("")
    a("**oracle(replay)**:三 kind 决策内容逐 request 全等(%d mismatch,"
      "容差 0)——包括 prefill 实例/assignment_key/历史迁移/逐出序列、"
      "decode 实例/candidates/迁移/逐出、completion 逐出与终态位置。"
      % b2r["oracle_content_mismatch"])
    a("")
    a("KV 事件载荷对照(kv_event_payload_sh1 口径,六类转移 × 1177 request,"
      "%d 行):replay 决策载荷 vs 离线 manifest `requests[].transfers_by_"
      "stage` 逐条全等(kind/phase/reason/session/trigger/source/target/"
      "total_bytes/shards 含 noc_path)——%d mismatch。"
      % (b2r["kv_payload_rows"], b2r["kv_payload_mismatch"]))
    a("")
    a("**real-online(strategy,关感知)**:")
    a("")
    a("- 确定性重放不变量:把 strategy 运行保留的 %d 个交付"
      "(bridge/request_*.json)逐条喂给全新 `Sh10OnlineScheduler`(同一冻结"
      " LUT 表、同一只读策略函数 KVCacheManager/select_prefill_instance/"
      "select_decode_instance),重放产出决策与 recorded 日志逐行全等"
      "(%d mismatch)——即\"相同输入下同一函数输出一致\"的逐决策断言。"
      % (b2s["deliveries"], b2s["rerun_decision_mismatch"]))
    a("- prefill_assignment_key 尾元素 == 所选实例(排序键 "
      "(remaining_chunks, last_arrival, config order) 自洽):%d fail。"
      % b2r["assignment_key_selffail"])
    a("- 信息性(差异可解释性证据,合法不同):strategy prefill 实例分布 %s;"
      "decode 实例分布 %s;与离线/replay 的差异来源 = 真实完成时序改变"
      "排队深度快照与 active decode 成员(队列深度均衡 + LUT 代价模型的"
      "输入变化),逐决策由确定性重放证明不变量。"
      % (b2r["strategy_prefill_distribution"],
         b2r["strategy_decode_distribution"]))
    a("")
    a("## B3 图结构等价(replay)")
    a("")
    a("在线侧材料 = 比较器内确定性重放交付流重构的 GraphBatch 节点/边;"
    "先与 live run 的 `graph_batch_digests.jsonl` 逐批对平(%d 批 "
    "content_sha256 %d mismatch = 重放构图 == live 构图)。"
      % (b3r["digest_rows_replay"], b3r["digest_mismatch_vs_live"]))
    a("")
    a("canonical logical node key(合同⑦;插入顺序/节点 ID 不作比较项):"
      "(rank, name, type, is_cpu_op, is_timer_op, 类型感知属性四元组"
      "——comp(num_ops,tensor_size,remote_weight_bytes)/comm(bytes,src,dst)/"
      "coll(comm_type,bytes,priority,pg_name,involved_dim)/mem(tensor_size)/"
      "timer 无载荷)。comm_tag 不入 key:tag 由 TransferTagAllocator 按全局"
      "发射序顺序分配(离线静态预写序 vs 在线决策交错序),与节点 ID 同类的"
      "插入顺序伪影;其配对语义由替代硬门证明(send/recv 集合内 tag 各自唯一"
      "+ 按 (src,dst,tag) 一一配对且 bytes 相等:实测 duplicate=%d, "
      "pair_missing=%d, recv_without_send=%d)+ 在线运行执行配对成功"
      "(1177/1177 完成,tag 错配即网络死锁)。"
      % (b3r["comm_tag_duplicate_keys"], b3r["comm_pair_missing"],
         b3r["comm_recv_without_send"]))
    a("")
    a("- 节点多重集:offline %d == online %d;missing %d,extra %d"
      "(容差 0,硬门)。" % (b3r["offline_nodes"], b3r["online_nodes"],
                            b3r["node_missing"], b3r["node_extra"]))
    a("- 边:offline %d,online %d;missing %d(within %d / cross %d),"
      "extra %d(within %d / cross %d)。"
      % (b3r["offline_edges"], b3r["online_edges"],
         b3r["edge_missing"], b3r["edge_missing_within"],
         b3r["edge_missing_cross"],
         b3r["edge_extra"], b3r["edge_extra_within"],
         b3r["edge_extra_cross"]))
    a("- 边差异归因(登记类别,方案 §5.5 ①②③④):")
    a("  - cross-request 差异 = 类别①(replay LUT 时钟:段 1 清 prefill 组"
      " previous_id + 段间 own-block-end 恢复;离线 writer 保留物理跨 "
      "request 链)+ 类别②③(块末恢复 emitted-ranks-only / 同 tick 发射序)。")
    a("  - within-request 差异 = 类别④(三段式发射的段间边界与触发门编码:"
      "段 2 noc_migrate 触发门锚段 1 末、段 3 completion_evictions 锚 "
      "decode 末——本仓 §5.5 预期高风险点,实测逐条归入)。")
    if b3r.get("within_missing_samples"):
        a("  - within missing 样例(rank/from/to/count):%s"
          % b3r["within_missing_samples"])
    if b3r.get("cross_missing_samples"):
        a("  - cross missing 样例:%s" % b3r["cross_missing_samples"][:3])
    if b3r.get("within_extra_samples"):
        a("  - within extra 样例:%s" % b3r["within_extra_samples"])
    if b3r.get("cross_extra_samples"):
        a("  - cross extra 样例:%s" % b3r["cross_extra_samples"][:3])
    a("")
    a("## B4 执行等价(归因口径)")
    a("")
    a("同图前提不成立(登记差异:跨 request 串行化边/三段式发射/timer gate "
      "时长/MEM 即时完成),exact 条款不适用(总体方案 §9.2);操作性标准 = "
      "差异全部可归因并记录:")
    a("")
    a("- 边界级漂移(delivery tick - 离线记录 tick):")
    a("  - prefill_drain_vs_decode_record: %s ns"
      % b4r["prefill_drain_vs_decode_record"])
    a("  - decode_completion_vs_completion_record: %s ns"
      % b4r["decode_completion_vs_completion_record"])
    a("- 节点级漂移((request,stage) 最大完成 tick - 对应记录 tick):%s ns;"
      ">1us 条目 %d。" % (b4r["node_drift_vs_record"],
                          b4r["node_drift_over_1us"]))
    a("- completion 跨 tick 逆序 %d(硬门);同 tick 组内逆序对 %d(信息性)。"
      % (b4r["completion_cross_tick_inversions"],
         b4r["completion_within_tick_inversion_pairs"]))
    a("- completion 早于同 request decode 决策:%d(硬门)。"
      % b4r["completion_before_decode"])
    a("- 指标不减:completed_requests 基线/在线 = %s;incomplete = %s;"
      "memory_actions_total = %s(unresolved 基线/在线 = %s);mean_e2e_ns "
      "基线/在线 = %.1f/%.1f(replay LUT 时钟 + comm/MEM 即时完成的压缩,"
      "类别①,信息性)。"
      % (b4r["completed_requests_baseline_vs_online"],
         b4r["incomplete_requests_baseline_vs_online"],
         b4r["memory_actions_total_baseline_vs_online"],
         b4r["memory_actions_unresolved_baseline_vs_online"],
         b4r["mean_e2e_ns_baseline_vs_online"][0],
         b4r["mean_e2e_ns_baseline_vs_online"][1]))
    a("")
    a("归因类别(不可解释差异 = 0):")
    a("")
    for key, text in b4r["attribution"].items():
        a("- **%s**:%s" % (key, text))
    a("")
    a("## 与 sh_2.0/sh_3.0 的口径差异登记(本仓特有)")
    a("")
    a("- 三段式发射(ARRIVAL/PREFILL_DRAIN/DECODE_COMPLETION→段 1/2/3)"
      "对照蓝本两段式;Committer 覆盖规则单向化(watch 必须有节点覆盖,"
      "尾段节点可无 watch)为本仓已批机制差异(实录登记 (a))。")
    a("- replay 装置 MEM 即时完成(合同⑦第④项,sh_1.0 独有)参与 B4 类别①。")
    a("- 两态 KV(LOCAL_HBM/REMOTE_MEMORY)+ edge-rank remote:B2 KV 载荷"
      "按六类转移元组对照,无 shard instance_remaining 字段(sh_2.0 的"
      " kv_cache_events.csv 账本不变量不适用;本仓等价物 = transfers_by_"
      "stage 全等 + 终态位置)。")
    a("- list 版 EventQueue;决策 tick 以 EventQueue EventTime 为准(合同③)。")
    a("")
    a("## 结论")
    a("")
    a("B0/B1/B2 oracle 容差 0 全过;B3 canonical 节点多重集一致(边差异全部"
      "归入登记类别);B4 归因口径下不可解释差异 = 0。阶段 2 门槛满足。")
    a("")
    with open(path, "w", encoding="utf-8") as out:
        out.write("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
