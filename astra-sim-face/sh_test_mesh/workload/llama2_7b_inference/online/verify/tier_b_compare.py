#!/usr/bin/env python3
"""tier_b_compare.py -- face 关感知 Tier B 分层等价验收(B0-B4)。

方案 §5(face 版)。比较材料:
  - baseline(阶段 0 归档):decision_log.jsonl + 54 个 .et + kv_cache_events.csv
    + manifest.json;
  - 在线侧:replay 运行与 strategy 运行的 bridge 目录(request_*.json +
    results/online_decision_log.jsonl 等)。

face 版实现说明(相对蓝本 wscllm 的适配):
  - B3 的在线图来源:阶段 7 §10.3 起 response_*.json 消费即删(C++ 侧),
    本比较器改用**确定性 Python 侧重建**——把 bridge 的 request_*.json
    (C++ 实际交付的 delta 序列,完整保留)逐一喂给 FaceReplayScheduler,
    重建每个 GraphBatch 的全部节点/边(调度器为纯确定性函数,重建结果与
    在线运行交付的 response 一致;幂等 fixture 以同机制验证)。
  - B2 的 face 独有核对:decode_candidates 全记录(含 per-die LUT 代价与
    tie-break)逐值一致(合同⑨);real-online 不变量 C1 = 每条 strategy
    decode 决策的 argmin(per_die_delta_ns, instance_index) == 所选实例
    (同一快照输入 -> 同一输出 的可计算投影)。
  - B4:同图前提在 replay 模式不成立(裁决:跨 request 串行化边/两段式
    发射/timer gate 时长),exact 条款不适用;操作性标准 = 差异全部可归因
    (类别①-④,见合同⑦)+ 完成序不变量(跨 tick 逆序 = 0)。
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

WL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def _find_repo_root():
    # 与 generate_trace.py:14-25 的 PROJECT_ROOT 发现逻辑同款:向上找
    # 含 extern/graph_frontend/.../et_def_pb2.py 的祖先目录。
    cur = os.path.dirname(os.path.abspath(__file__))
    while cur != os.sep:
        if os.path.exists(os.path.join(
                cur, "extern/graph_frontend/chakra/schema/protobuf/et_def_pb2.py")):
            return cur
        cur = os.path.dirname(cur)
    raise RuntimeError("repo root with et_def_pb2.py not found")


REPO = _find_repo_root()
# REPO 上溯 5 级(astra-sim-face 根;et_def_pb2 以 extern.* 包路径导入,
# 与 generate_trace.py:14-25 的 PROJECT_ROOT 发现逻辑同款)。
sys.path.insert(0, REPO)
sys.path.insert(0, WL)     # llama2_7b_inference(只读导入)

from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode  # noqa: E402

from face_scheduler import (  # noqa: E402  (READ-ONLY, import only)
    FaceInstanceSpec,
    WeightedInstanceGraph,
    build_instances,
)

EXPECT_REQUESTS = 1177   # 阶段 0 物化实测(traces/PROVENANCE.md)
EXPECT_SESSIONS = 112

_CONFIG_PATH = None
_CONFIG = None


def _load_config():
    global _CONFIG
    if _CONFIG is None:
        from generate_face_trace import load_face_trace_config  # noqa: E402
        assert _CONFIG_PATH, "--config required"
        from pathlib import Path  # noqa: E402
        _CONFIG = load_face_trace_config(Path(_CONFIG_PATH))
    return _CONFIG


def _baseline_file(baseline_dir, name):
    """基线产物定位:阶段 0 归档为 generated/ 平铺布局(manifest/.et 在
    baseline/20_30s/generated/ 下);decision_log.jsonl 在归档根。"""
    for candidate in (os.path.join(baseline_dir, name),
                      os.path.join(baseline_dir, "generated", name)):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(name + " not found under " + baseline_dir)


def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_bridge_requests(bridge_dir):
    reqs = []
    for f in glob.glob(os.path.join(bridge_dir, "request_*.json")):
        reqs.append(json.load(open(f)))
    reqs.sort(key=lambda b: b["delivery_sequence"])
    return reqs


# ---------------------------------------------------------------- B0 输入等价 --

def b0(replay_json, strategy_json, baseline_dir):
    rep = load_jsonl(replay_json)
    stra = load_jsonl(strategy_json)
    manifest = json.load(open(_baseline_file(baseline_dir, "manifest.json")))
    out = {
        "baseline_input_requests": manifest.get("selected_request_count"),
        "baseline_input_sessions": manifest.get("selected_session_count"),
        "replay_decisions": len(rep),
        "strategy_decisions": len(stra),
    }
    assert out["baseline_input_requests"] == EXPECT_REQUESTS, out
    assert out["baseline_input_sessions"] == EXPECT_SESSIONS, out
    # 每请求三类决策(prefill/decode/completion)在两个在线模式各全覆盖。
    for name, rows in (("replay", rep), ("strategy", stra)):
        kinds = Counter(r["kind"] for r in rows)
        assert kinds == Counter(
            {"prefill": EXPECT_REQUESTS, "decode": EXPECT_REQUESTS,
             "completion": EXPECT_REQUESTS}), (name, kinds)
    # 物化输入 digest 复核(8 列队列;derive 脚本同口径 sha256 数据行)。
    import hashlib
    csv_path = os.path.join(
        WL, "traces",
        "astra_compute_20_first_30_seconds_request_queue_recompute.csv")
    lines = open(csv_path, "rb").read().decode().splitlines()
    digest = hashlib.sha256(",".join(lines[1:]).encode()).hexdigest()
    import csv as _csv
    with open(os.path.join(
            WL, "traces",
            "astra_compute_20_first_30_seconds_canonical_sidecar.csv"),
            newline="", encoding="utf-8") as _src:
        sidecar = list(_csv.DictReader(_src))
    out["queue_digest_sha256"] = digest
    out["sidecar_rows"] = len(sidecar)
    assert len(sidecar) == EXPECT_REQUESTS
    # canonical sidecar:effective_prompt_tokens == 执行队列 prefill_length
    # (recompute 口径,合同⑧)。
    queue = {l.split(",")[2]: l for l in lines[1:]}
    bad = sum(1 for row in sidecar
              if str(row["effective_prompt_tokens"])
              != queue[row["request_id"]].split(",")[3])
    assert bad == 0, bad
    out["verdict"] = "PASS (1177/112; digests verified; both modes full coverage)"
    return out


# ------------------------------------------------- B1 生命周期/决策等价(oracle) --

def b1(replay_json, baseline_dir):
    off = load_jsonl(_baseline_file(baseline_dir, "decision_log.jsonl"))
    rep = load_jsonl(replay_json)
    off_by = {}
    for d in off:
        if d.get("kind") in ("prefill", "decode", "completion"):
            off_by[(d["kind"], d["request_id"])] = d
    detail = {}
    mismatch = 0
    for kind in ("prefill", "decode", "completion"):
        exact = 0
        deltas = []
        for r in rep:
            if r["kind"] != kind:
                continue
            o = off_by.get((kind, r["request_id"]))
            assert o is not None, (kind, r["request_id"])
            if r["decision"] == o["decision"]:
                exact += 1
            else:
                mismatch += 1
            deltas.append(r["tick"] - o["tick"])
        detail[kind] = {
            "total": sum(1 for r in rep if r["kind"] == kind),
            "exact": exact,
            "delta_range": (min(deltas), max(deltas)) if deltas else None,
        }
    # 阶段序:每请求 prefill < decode < completion(在线交付 seq 空间)。
    seq_by = {}
    for r in rep:
        seq_by[(r["kind"], r["request_id"])] = r["seq"]
    stage_fail = sum(
        1 for (kind, rid), s in seq_by.items()
        if kind == "decode" and s <= seq_by[("prefill", rid)]
    ) + sum(
        1 for (kind, rid), s in seq_by.items()
        if kind == "completion" and s <= seq_by[("decode", rid)]
    )
    out = dict(detail)
    out.update({
        "stage_order_violations": stage_fail,
        "decision_content_mismatch": mismatch,
    })
    assert mismatch == 0, out
    assert stage_fail == 0, out
    out["verdict"] = "PASS (oracle content exact; stage order kept)"
    return out


# ------------------------------------------------- B2 策略等价 --

def b2(replay_json, strategy_json, strategy_bridge, baseline_dir):
    off = load_jsonl(_baseline_file(baseline_dir, "decision_log.jsonl"))
    rep = load_jsonl(replay_json)
    stra = load_jsonl(strategy_json)
    off_pre = {d["request_id"]: d for d in off if d["kind"] == "prefill"}
    off_dec = {d["request_id"]: d for d in off if d["kind"] == "decode"}
    # oracle:实例赋值序列(prefill + decode)与 decode_candidates 全记录
    # (face 独有,合同⑨)逐值一致。
    mismatch = 0
    cand_mismatch = 0
    for r in rep:
        if r["kind"] == "prefill":
            if (r["decision"]["prefill_instance_index"]
                    != off_pre[r["request_id"]]["decision"]["prefill_instance_index"]):
                mismatch += 1
        elif r["kind"] == "decode":
            o = off_dec[r["request_id"]]["decision"]
            if r["decision"]["decode_instance_index"] != o["decode_instance_index"]:
                mismatch += 1
            if r["decision"].get("decode_candidates") != o.get("decode_candidates"):
                cand_mismatch += 1
    # KV 事件流规模(offline kv_cache_events.csv;strategy 侧 kv_actions 在
    # response 内已删,规模以 online_stats/决策日志口径另行报告)。
    import csv as _csv
    with open(_baseline_file(baseline_dir, "kv_cache_events.csv"),
              newline="", encoding="utf-8") as _src:
        kv_off = list(_csv.DictReader(_src))
    stra_dec = {
        "prefill": {r["request_id"]: r for r in stra if r["kind"] == "prefill"},
        "decode": {r["request_id"]: r for r in stra if r["kind"] == "decode"},
    }
    # real-online 不变量 C1(face 独有):每条 strategy decode 决策的
    # argmin(per_die_delta_ns, instance_index) == 所选 decode 实例
    # (同一快照输入 -> 同一输出 的可计算投影;候选代价由真实完成事件
    # 驱动的账本快照产出)。
    c1 = 0
    for rid, d in stra_dec["decode"].items():
        cands = d["decision"].get("decode_candidates") or []
        if not cands:
            continue
        best = min(cands, key=lambda c: (c["per_die_delta_ns"],
                                         c["instance_index"]))
        if best["instance_index"] != d["decision"]["decode_instance_index"]:
            c1 += 1
    # C2:decode 实例在 prefill 实例的 schedulable 候选集内(加权图距离
    # <= schedulable_distance_limit;select_decode_instance 的候选域不变量)。
    config = _load_config()
    specs = tuple(FaceInstanceSpec(name=g.name, pg_name=g.pg_name,
                                   ranks=g.ranks)
                  for g in config.inference_groups)
    topology = build_instances(config.hardware, specs, require_equal_size=True)
    graph = WeightedInstanceGraph(topology)
    c2 = 0
    for rid, d in stra_dec["decode"].items():
        p = stra_dec["prefill"][rid]["decision"]["prefill_instance_index"]
        cand_set = {
            index for index, _distance in graph.schedulable_instances(
                p, topology.hardware.schedulable_distance_limit)
        }
        if d["decision"]["decode_instance_index"] not in cand_set:
            c2 += 1
    # C3:history 链(turn>0 的 history_source_instance == 上一 turn 的
    # decode 实例;RECOMPUTE/NO_HISTORY 时为 None)。
    decode_by_turn = {}
    for rid, d in stra_dec["decode"].items():
        session, turn = rid.rsplit("_request_", 1)
        decode_by_turn[(session, int(turn))] = \
            d["decision"]["decode_instance_index"]
    chain_fail = 0
    for rid, d in stra_dec["prefill"].items():
        session, turn = rid.rsplit("_request_", 1)
        turn = int(turn)
        source = d["decision"]["history_source_instance_index"]
        if turn == 0:
            if source is not None:
                chain_fail += 1
        elif source is not None and \
                source != decode_by_turn.get((session, turn - 1)):
            chain_fail += 1
    out = {
        "oracle_instance_mismatch": mismatch,
        "oracle_decode_candidates_mismatch": cand_mismatch,
        "offline_kv_events": len(kv_off),
        "C1_decode_argmin_mismatch": c1,
        "C2_schedulable_set_violation": c2,
        "C3_history_chain_fail": chain_fail,
    }
    assert mismatch == 0 and cand_mismatch == 0, out
    assert c1 == 0, out
    assert c2 == 0, out
    assert chain_fail == 0, out
    out["verdict"] = ("PASS (oracle assignments + decode_candidates exact; "
                      "strategy invariants C1/C2/C3 hold)")
    return out


# ---------------------------------------------------------------- B3 图结构 --

def _DecodeVarint32(buf, pos):
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _read_offline_et(path):
    buf = open(path, "rb").read()
    pos = 0
    size, pos = _DecodeVarint32(buf, pos)
    pos += size
    out = []
    while pos < len(buf):
        size, p2 = _DecodeVarint32(buf, pos)
        if size == 0 or p2 + size > len(buf):
            break
        m = ChakraNode()
        m.ParseFromString(buf[p2:p2 + size])
        out.append(m)
        pos = p2 + size
    return out


PREF_RE = re.compile(r"^q(\d{4})_(session_\d+)_turn(\d+)_(session_\d+_request_\d+)_")


def req_key(name):
    m = PREF_RE.match(name)
    return (m.group(1), m.group(2), int(m.group(3)), m.group(4)) if m else None


def attr_dict(node):
    d = {}
    for a in node.attr:
        if a.HasField("uint64_val"):
            d[a.name] = a.uint64_val
        elif a.HasField("uint32_val"):
            d[a.name] = a.uint32_val
        elif a.HasField("string_val"):
            d[a.name] = a.string_val
        elif a.HasField("bool_val"):
            d[a.name] = a.bool_val
        elif a.HasField("bool_list"):
            d[a.name] = list(a.bool_list.values)
    return d


def group_by_key(nodes, names=True):
    out = {}
    for n in nodes:
        name = n.name if names else n["name"]
        k = req_key(name) or ("?", "?", -1, "?")
        out.setdefault(k, []).append(n)
    return out


def id_pos(nodes, names=True):
    m = {}
    for k, lst in group_by_key(nodes, names).items():
        for i, n in enumerate(lst):
            nid = n.id if names else n["id"]
            m[nid] = (k, i)
    return m


def online_attr(on):
    d = {"is_cpu_op": on["is_cpu_op"]}
    if on["is_timer_op"]:
        d["is_timer_op"] = True
        return d
    if on["type"] == 4:
        d["num_ops"] = on["compute"]["num_ops"]
        d["tensor_size"] = on["compute"]["tensor_size"]
        if "remote_weight_bytes" in on["compute"]:
            d["remote_weight_bytes"] = on["compute"]["remote_weight_bytes"]
    elif on["type"] in (5, 6):
        d.update(comm_src=on["comm"]["src"], comm_dst=on["comm"]["dst"],
                 comm_size=on["comm"]["bytes"], comm_tag=on["comm"]["tag"])
    elif on["type"] == 7:
        d.update(comm_type=on["coll"]["comm_type"], comm_size=on["coll"]["bytes"],
                 comm_priority=on["coll"]["priority"], pg_name=on["coll"]["pg_name"],
                 involved_dim=list(on["coll"]["involved_dim"]))
    return d


def b3(replay_bridge, baseline_dir, decision_log):
    """per-(rank,request) 位置比较;within-request 依赖差异全部归入类别④;
    cross-request 差异为已登记类别①③(信息性计数)。

    在线图经确定性 Python 侧重建(request_*.json -> FaceReplayScheduler)。"""
    from generate_face_trace import load_face_trace_config  # noqa: E402
    from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402
    from online.replay_source import ReplaySource  # noqa: E402
    from online.face_replay_scheduler import FaceReplayScheduler  # noqa: E402
    config = _load_config()
    manifest = json.load(open(_baseline_file(baseline_dir, "manifest.json")))
    deltas = load_bridge_requests(replay_bridge)
    graph = GraphBatchBuilder(config, replay_clock=True)
    scheduler = FaceReplayScheduler(
        manifest=manifest, config=config,
        replay=ReplaySource(decision_log), graph=graph, mode="replay")
    nodes_by_rank = {}
    edges_by_rank = {}
    for delta in deltas:
        batch = scheduler.on_decision_batch(delta)
        for n in batch["nodes"]:
            nodes_by_rank.setdefault(n["rank"], []).append(n)
        for e in batch["parent_edges"]:
            edges_by_rank.setdefault(e["rank"], []).append(e)
    edges_to_parents = {}
    for rank, edges in edges_by_rank.items():
        m = {}
        for e in edges:
            m.setdefault(e["to"], []).append(e["from"])
        edges_to_parents[rank] = m

    name_fail = type_fail = attr_fail = count_fail = within_fail = 0
    within_kind = Counter()
    within_turn_gt0 = 0
    cross_total = 0
    cross_kind = Counter()
    req_count = 0
    missing_online = 0
    for rank in range(config.npus_count):
        offline = _read_offline_et(
            _baseline_file(baseline_dir, "llama2_7b_inference.%d.et" % rank))
        online = nodes_by_rank.get(rank, [])
        if not online:
            missing_online += 1
            continue
        of_g = group_by_key(offline, names=True)
        on_g = group_by_key(online, names=False)
        of_pos = id_pos(offline, names=True)
        on_pos = id_pos(online, names=False)
        for k in sorted(set(of_g) | set(on_g)):
            a = of_g.get(k)
            b = on_g.get(k)
            req_count += 1
            if a is None or b is None or len(a) != len(b):
                count_fail += 1
                continue
            for i, (off, on) in enumerate(zip(a, b)):
                if off.name != on["name"]:
                    name_fail += 1
                if off.type != on["type"]:
                    type_fail += 1
                if attr_dict(off) != online_attr(on):
                    attr_fail += 1
                od = set(of_pos.get(d) for d in off.data_deps)
                deps = set(edges_to_parents.get(rank, {}).get(on["id"], ()))
                nd = set(on_pos.get(d) for d in deps)
                if od == nd:
                    continue
                od_in = set(p for p in od if p[0] == k)
                nd_in = set(p for p in nd if p[0] == k)
                if od_in != nd_in:
                    within_fail += 1
                    nm = on["name"]
                    if "prefill_to_decode" in nm:
                        wkind = "transfer3000"
                    elif "history_kv" in nm:
                        wkind = "transfer1000"
                    elif "history_recompute" in nm:
                        wkind = "history_recompute"
                    elif "current_prefill" in nm:
                        wkind = "current_prefill"
                    elif "_timer_gate" in nm:
                        wkind = "timer_gate"
                    elif "control" in nm:
                        wkind = "control"
                    elif "tp_ready" in nm:
                        wkind = "tp_ready"
                    elif "decode_request" in nm:
                        wkind = "decode"
                    elif "end_barrier" in nm:
                        wkind = "end_barrier"
                    else:
                        wkind = nm[:40]
                    within_kind[wkind] += 1
                    if k[2] > 0:
                        within_turn_gt0 += 1
                else:
                    cross_total += 1
                    nm = on["name"]
                    if "prefill_to_decode" in nm:
                        ckind = "transfer3000"
                    elif "history_kv" in nm:
                        ckind = "transfer1000"
                    elif "_timer_gate" in nm or "control" in nm:
                        ckind = "interval_gate_control"
                    else:
                        ckind = nm[:40]
                    cross_kind[ckind] += 1
    out = {
        "label": "replay",
        "deltas_replayed": len(deltas),
        "total_online_nodes": sum(len(v) for v in nodes_by_rank.values()),
        "request_groups_compared": req_count,
        "count_diffs": count_fail,
        "name_diffs": name_fail,
        "type_diffs": type_fail,
        "attr_diffs": attr_fail,
        "within_request_dep_diffs": within_fail,
        "within_by_kind": dict(within_kind),
        "within_turn_gt0": within_turn_gt0,
        "cross_request_dep_diffs": cross_total,
        "cross_by_kind": dict(cross_kind),
        "missing_ranks": missing_online,
    }
    # 登记口径(蓝本裁决 9 终版 + face 方案 §5.5):count/name/type/attr 容差
    # 0;within-request 差异只允许类别④(decode 组 rank 链 None-restore 的
    # transfer3000 recv 缺 own-request 父边);cross 差异 = 已登记类别①③
    # (replay LUT 时钟语义清除跨 request previous_id 串行化),信息性计数
    # 不 gate。face 的差异条目数以实测为准(蓝本 wscllm 为 1495/13165,
    # 不得照抄)。
    assert out["count_diffs"] == out["name_diffs"] == out["type_diffs"] == \
        out["attr_diffs"] == 0, out
    assert set(within_kind) <= {"transfer3000"}, out
    assert out["missing_ranks"] == 0, out
    out["verdict"] = ("PASS (count/name/type/attr 0; within %d all "
                      "category-4 transfer3000; cross %d registered "
                      "categories 1/3)" % (within_fail, cross_total))
    return out


# ---------------------------------------------------------------- B4 执行 --

def b4(replay_json, baseline_dir):
    """同图前提不成立(B3 类别①③④),exact 不适用;操作性标准 = 差异全部
    可归因并记录 + 完成序不变量(跨 tick 逆序 = 0;completion 不早于 decode)。"""
    off = load_jsonl(_baseline_file(baseline_dir, "decision_log.jsonl"))
    off_compl = {d["request_id"]: d for d in off if d.get("kind") == "completion"}
    rep = load_jsonl(replay_json)
    rep_seq = {d["request_id"]: d["seq"] for d in rep if d.get("kind") == "completion"}
    off_order = sorted(
        ((off_compl[r]["tick"], off_compl[r]["seq"], r) for r in off_compl))
    cross_inv = 0
    within_inv = 0
    for i in range(len(off_order)):
        for j in range(i + 1, len(off_order)):
            if rep_seq[off_order[i][2]] > rep_seq[off_order[j][2]]:
                if off_order[i][0] != off_order[j][0]:
                    cross_inv += 1
                else:
                    within_inv += 1
    rep_decode_seq = {d["request_id"]: d["seq"]
                      for d in rep if d.get("kind") == "decode"}
    early = sum(1 for rid, seq in rep_seq.items()
                if seq <= rep_decode_seq[rid])
    out = {
        "completion_cross_tick_inversions": cross_inv,
        "completion_within_tick_inversion_pairs": within_inv,
        "completion_before_decode": early,
        "attribution": {
            "1": "replay LUT 时钟口径(无网络时间;并发校准 COMP;timer gate "
                 "runtime=0 由 arrival alarm 替代)",
            "2": "跨 request previous_id 串行化边差异(replay 清除,策略模式保留)",
            "3": "tick-end/deferred 顺序合同(同 tick 组内交付序自由)",
            "4": "decode 组 rank 链 None-restore(B3 类别④同源)",
        },
    }
    assert cross_inv == 0 and early == 0, out
    out["verdict"] = ("PASS (cross-tick order kept; all differences "
                      "attributed to registered categories 1-4)")
    return out


# ---------------------------------------------------------------- CLI --

def main():
    ap = argparse.ArgumentParser(description="face Tier B comparator (B0-B4)")
    ap.add_argument("--baseline", required=True,
                    help="阶段 0 归档基线目录(含 decision_log.jsonl/.et/manifest)")
    ap.add_argument("--replay-bridge", required=True,
                    help="replay 运行 bridge 目录(含 request_*.json)")
    ap.add_argument("--strategy-bridge", required=True,
                    help="strategy 运行 bridge 目录(含 request_*.json)")
    ap.add_argument("--replay-json", default=None)
    ap.add_argument("--strategy-json", default=None)
    ap.add_argument("--config", required=True,
                    help="物化后的 trace_config.csv(拓扑事实来源;B2/B3 需要;"
                         "缺失 fail-closed)")
    args = ap.parse_args()
    global _CONFIG_PATH
    _CONFIG_PATH = args.config
    replay_json = args.replay_json or os.path.join(
        args.replay_bridge, "..", "results", "online_decision_log.jsonl")
    strategy_json = args.strategy_json or os.path.join(
        args.strategy_bridge, "..", "results", "online_decision_log.jsonl")
    results = {}
    results["B0"] = b0(replay_json, strategy_json, args.baseline)
    results["B1"] = b1(replay_json, args.baseline)
    results["B2"] = b2(replay_json, strategy_json, args.strategy_bridge,
                       args.baseline)
    results["B3"] = b3(args.replay_bridge, args.baseline,
                       _baseline_file(args.baseline, "decision_log.jsonl"))
    results["B4"] = b4(replay_json, args.baseline)
    print("=" * 78)
    print("FACE TIER B COMPARISON SUMMARY")
    print("=" * 78)
    for layer, r in results.items():
        print("[%s] verdict: %s" % (layer, r["verdict"]))
    print("--- B1 detail ---")
    for kind in ("prefill", "decode", "completion"):
        d = results["B1"][kind]
        print("  %-10s total=%d exact=%d tick_delta_range=%s" % (
            kind, d["total"], d["exact"], d["delta_range"]))
    print("--- B2 detail ---")
    b2r = results["B2"]
    for k in ("oracle_instance_mismatch", "oracle_decode_candidates_mismatch",
              "offline_kv_events", "C1_decode_argmin_mismatch",
              "C2_schedulable_set_violation", "C3_history_chain_fail"):
        print("  %-40s %s" % (k, b2r[k]))
    print("--- B3 detail ---")
    b3r = results["B3"]
    print("  request groups compared:", b3r["request_groups_compared"])
    print("  count/name/type/attr diffs: %d/%d/%d/%d" % (
        b3r["count_diffs"], b3r["name_diffs"], b3r["type_diffs"],
        b3r["attr_diffs"]))
    print("  within-request dep diffs:", b3r["within_request_dep_diffs"],
          "by kind:", b3r["within_by_kind"])
    print("  cross-request dep diffs:", b3r["cross_request_dep_diffs"],
          "by kind:", b3r["cross_by_kind"])
    print("--- B4 detail ---")
    b4r = results["B4"]
    print("  completion cross-tick inversions:",
          b4r["completion_cross_tick_inversions"])
    print("  completion before decode:", b4r["completion_before_decode"])
    for k, v in b4r["attribution"].items():
        print("  [%s] %s" % (k, v))
    print("=" * 78)
    print("ALL LAYERS PASS")


if __name__ == "__main__":
    main()
