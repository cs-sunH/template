#!/usr/bin/env python3
"""tier_b_compare.py — Tier B(B0-B4)集成比较器(阶段 2 交付物)。

合同⑦验证方法规定的比较器:逐层比较输出差异报告。基线侧材料 = 调用方
按方案文档 §3 步骤 0-1 物化输入并运行静态 ET 后产出的基线目录
(decision_log.jsonl + 54 .et + kv_cache_events.csv + manifest.json;本仓库
不预置,经 --baseline / --legacy-baseline 显式传入,缺失 fail-closed);
在线侧 = 官方 runner 运行目录(bridge/:request_*.json、response_*.json、
online_decision_log.jsonl)。拓扑事实(config.inference_groups/hardware/
npus_count)取自调用方物化配置(--config,缺失 fail-closed)。

分层:
  B0 输入等价      请求/session 计数、决策日志行数、prefill 决策内容 digest、
                   KV 事件计数与离线同源核对(容差 0)。
  B1 生命周期/决策 oracle/replay 决策 tick 对离线 decision_log 的 delta 直方图
                   (容差 0,偏差按已登记类别归因:③ tick-end/deferred 顺序合同);
                   逆序/提前/阶段序违规检查。
  B2 策略等价      oracle:决策内容全等 + 与离线 kv_cache_events.csv 交叉核对
                   (history/prefill_decode 迁移字节/eviction/completion/KV 账本
                   不变量);real-online(strategy):C1 静态路由 / C2 prefill
                   argmin / C3 decode 准入深度逐决策不变量。
  B3 图结构等价    replay 在线发射图 vs 离线 .et per-(rank,request) 位置比较:
                   count/name/type/attr 容差 0;within-request 依赖差异全部归入
                   已登记类别 ④(decode 组 rank 链 None-restore,5abaf23 为
                   LUT 时钟稳定性的必要机制,两轮决定性验证见 phase2_status.md);
                   cross-request 边 = 0 为刻意差异(类别①)。
  B4 执行          同图前提不成立,exact 不适用;操作性标准 = 差异全部可归因并
                   记录(类别①②③④);本层输出归因清单与可计算的不变量
                   (replay 完成序 vs LUT 序、无提前决策)。
  B-LEGACY         阶段 7 §10.6 legacy 第二变体(real-online)对照:在线 legacy
                   运行 vs 离线 legacy baseline(20_30s_legacy)。硬门只验策略
                   结构性不变量(行数/覆盖/阶段序/字段形状/legacy 常量字段/
                   history 字节确定性/terminal 确定性/链内一致性/C1 静态路由/
                   allocator 终值),顺序敏感字段为信息性差异报告(合同⑦)。
                   --legacy-run <run_dir> 给定才运行。

用法:
  python3 tier_b_compare.py \
    --replay-bridge <run_dir>/bridge \
    --strategy-bridge <run_dir>/bridge \
    --baseline <物化基线目录(方案文档 §3 步骤 0-1)> --config <物化配置>

退出码:全部断言成立 → 0;任何记录数值复现失败 → 1(比较器不得放宽掩盖失败)。
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter

WL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
# 仓库根:…/sh_test_mesh/workload/llama2_7b_inference 上溯三级
# (llama2_7b_inference -> workload -> sh_test_mesh -> 仓库根)
ROOT = os.path.realpath(os.path.join(WL, "..", "..", ".."))
# request-neutral(裸仓库收尾):不预置/不默认绑定任何物化基线目录与物化配置。
# 基线由调用方按方案文档 §3 步骤 0-1 物化后经 --baseline / --legacy-baseline
# 显式传入,拓扑配置经 --config 显式传入;缺失均 fail-closed(见 main)。

sys.path.insert(0, ROOT)      # extern/…(protobuf schema)
sys.path.insert(0, WL)        # llama2_7b_inference(generate_* / wsc_llm_scheduler,只读)

from google.protobuf.internal.decoder import _DecodeVarint32  # noqa: E402
from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode  # noqa: E402
from generate_wsc_llm_trace import load_wsc_llm_trace_config  # noqa: E402

# request-neutral:不在模块导入期加载 checked-in 配置(其 request_queue_csv
# 为占位符,加载即 fail-closed)。拓扑事实按需经 --config 显式传入后
# 惰性加载(调用方物化环境,队列文件已物化,加载合法)。
_CONFIG_PATH = None
_config = None


def _load_config():
    global _config
    if _config is None:
        if _CONFIG_PATH is None:
            print("missing --config: 拓扑事实(config.inference_groups/hardware/"
                  "npus_count)需物化配置;请按方案文档 wscllm仓库改造详细执行方案.md "
                  "§3 步骤 0-1 物化输入后显式传入", file=sys.stderr)
            sys.exit(1)
        _config = load_wsc_llm_trace_config(_CONFIG_PATH)
    return _config

# ------------------------------------------------------------------- helpers --


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_bridge_responses(bridge_dir):
    resp = []
    for f in glob.glob(os.path.join(bridge_dir, "response_*.json")):
        resp.append(json.load(open(f)))
    resp.sort(key=lambda b: b["source_delivery_sequence"])
    return resp


def load_bridge_requests(bridge_dir):
    reqs = []
    for f in glob.glob(os.path.join(bridge_dir, "request_*.json")):
        reqs.append(json.load(open(f)))
    reqs.sort(key=lambda b: b["delivery_sequence"])
    return reqs


# ---------------------------------------------------------------- B0 输入等价 --


def b0(replay_json, strategy_json, baseline_dir):
    out = {}
    off = load_jsonl(os.path.join(baseline_dir, "decision_log.jsonl"))
    off_dec = [d for d in off if d.get("kind") in ("prefill", "decode", "completion")]
    off_requests = {d["request_id"] for d in off_dec}
    manifest = json.load(open(os.path.join(baseline_dir, "manifest.json")))

    rep = load_jsonl(replay_json)
    stra = load_jsonl(strategy_json)
    out["offline_decision_lines"] = len(off)
    out["online_replay_lines"] = len(rep)
    out["online_strategy_lines"] = len(stra)
    out["offline_requests"] = len(off_requests)
    out["online_replay_requests"] = len({d["request_id"] for d in rep})
    out["online_strategy_requests"] = len({d["request_id"] for d in stra})
    out["manifest_requests"] = manifest.get("selected_request_count")
    out["manifest_sessions"] = manifest.get("selected_session_count")
    assert out["manifest_requests"] == 1177 and out["manifest_sessions"] == 112, out
    # prefill 决策内容 digest(在线 replay 与离线同源决策序列)。seq 为记录
    # 编号约定差异(离线 per-kind 0-based,在线 delivery 1-based),不参与
    # 内容等价;tick 与 decision 内容必须全等(B1 层单独比较 tick 分布)。
    off_prefill = {d["request_id"]: d for d in off_dec if d["kind"] == "prefill"}
    rep_prefill = {d["request_id"]: d for d in rep if d["kind"] == "prefill"}
    digest_mismatch = 0
    for rid, d in off_prefill.items():
        r = rep_prefill.get(rid)
        if r is None or r.get("decision") != d.get("decision") \
                or r.get("tick") != d.get("tick"):
            digest_mismatch += 1
    out["prefill_digest_mismatch"] = digest_mismatch
    # KV 事件行数(离线 csv 7932;strategy kv_actions 5935,replay 无策略动作)
    rows = list(csv.DictReader(open(os.path.join(baseline_dir, "kv_cache_events.csv"))))
    out["offline_kv_rows"] = len(rows)
    assert out["offline_decision_lines"] == 214450, out
    assert out["online_replay_lines"] == 3531, out
    assert out["online_strategy_lines"] == 3531, out
    assert out["offline_requests"] == out["online_replay_requests"] == \
        out["online_strategy_requests"] == 1177, out
    assert out["prefill_digest_mismatch"] == 0, out
    out["verdict"] = "PASS"
    return out


# ------------------------------------------- B1 生命周期/决策(oracle/replay) --


def b1(replay_json, baseline_dir):
    """oracle = 离线 decision_log.jsonl;容差 0。偏差按已登记类别归因。"""
    off = load_jsonl(os.path.join(baseline_dir, "decision_log.jsonl"))
    off_dec = {k: {d["request_id"]: d for d in off if d.get("kind") == k}
               for k in ("prefill", "decode", "completion")}
    rep = load_jsonl(replay_json)
    rep_dec = {k: {d["request_id"]: d for d in rep if d.get("kind") == k}
               for k in ("prefill", "decode", "completion")}
    out = {}
    for kind in ("prefill", "decode", "completion"):
        deltas = Counter()
        positive = []
        for rid in sorted(off_dec[kind]):
            off_tick = off_dec[kind][rid]["tick"]
            on_tick = rep_dec[kind][rid]["tick"]
            deltas[on_tick - off_tick] += 1
            if on_tick - off_tick > 0:
                positive.append((rid, on_tick - off_tick))
        exact = deltas.get(0, 0)
        total = sum(deltas.values())
        neg = sum(v for k, v in deltas.items() if k < 0)
        pos = sum(v for k, v in deltas.items() if k > 0)
        out[kind] = {"total": total, "exact": exact, "negative": neg,
                     "positive": pos, "delta_range": [min(deltas), max(deltas)],
                     "positive_samples": positive[:3]}
        assert total == 1177, (kind, total)
    # 已登记 B1 口径:prefill 1177/1177 精确;decode 23 精确 + 1153 负偏差 +
    # 1 正偏差(共 1154 落在 [-12,+1],唯一正偏差 session_41_request_0,
    # 类别③);completion 1177 全部落在 [-17,-1](类别①),0 正偏差。
    assert out["prefill"]["exact"] == 1177 and out["prefill"]["positive"] == 0, out["prefill"]
    assert out["decode"]["exact"] == 23 and out["decode"]["negative"] == 1153 \
        and out["decode"]["positive"] == 1, out["decode"]
    assert out["completion"]["exact"] == 0 and out["completion"]["positive"] == 0 \
        and out["completion"]["negative"] == 1177, out["completion"]
    assert min(out["decode"]["delta_range"]) >= -12 and max(out["decode"]["delta_range"]) <= 1
    assert min(out["completion"]["delta_range"]) >= -17 and max(out["completion"]["delta_range"]) <= -1
    # 逆序检查:离线事件序列(同 tick 内 completion < decode < prefill,
    # 与 wscllm_diff_sum.py 的 (tick, priority, seq) 一致)与在线交付序
    # (delivery seq)比对——跨 tick 逆序 = 0(硬门);同 tick 等值组内顺序
    # 自由(③ tick-end/deferred 顺序合同),组内逆序对数为信息性计数。
    rep_seq = {(d["kind"], d["request_id"]): d["seq"]
               for d in rep}
    prio = {"completion": 0, "decode": 1, "prefill": 2}
    ev = sorted((off_dec[k][rid]["tick"], prio[k], off_dec[k][rid]["seq"], k, rid)
                for k in ("prefill", "decode", "completion") for rid in off_dec[k])
    cross_tick_inv = 0
    within_tick_inv = 0
    for i in range(len(ev)):
        for j in range(i + 1, len(ev)):
            if rep_seq[(ev[i][3], ev[i][4])] > rep_seq[(ev[j][3], ev[j][4])]:
                if ev[i][0] != ev[j][0]:
                    cross_tick_inv += 1
                else:
                    within_tick_inv += 1
    out["cross_tick_inversions"] = cross_tick_inv
    out["within_tick_inversion_pairs"] = within_tick_inv
    assert cross_tick_inv == 0, out
    # 阶段序:同一 request 的 prefill.tick <= decode.tick <= completion.tick
    stage_violations = 0
    for rid in off_dec["prefill"]:
        if not (off_dec["prefill"][rid]["tick"] <= off_dec["decode"][rid]["tick"]
                <= off_dec["completion"][rid]["tick"]):
            stage_violations += 1
    out["stage_order_violations"] = stage_violations
    assert stage_violations == 0, out
    out["verdict"] = "PASS"
    return out


# ----------------------------------------------------------------- B2 策略 --


def _kv_ledger_invariants(rows, label):
    """after-before == sign×shard_bytes;sum(shards)==total_bytes;
    instance_remaining 长度 == 6(TP rank)。"""
    def _as_list(v):
        if v is None or v == "":
            return []
        return v if isinstance(v, list) else json.loads(v)

    fails = 0
    for i, r in enumerate(rows):
        shards = _as_list(r["shard_bytes"])
        before = _as_list(r["instance_remaining_before_bytes"])
        after = _as_list(r["instance_remaining_after_bytes"])
        total = int(r["total_bytes"])
        if shards:
            if sum(shards) != total:
                fails += 1
                continue
            if len(before) != 6 or len(after) != 6 or len(shards) != 6:
                fails += 1
                continue
            # 状态未变行(retain_complete 等:shard 为信息性,remaining 不变)
            # 不参与 diff 校验;其余行统一校验每个 rank 的 diff 绝对值 ==
            # shard(sign 语义因 noc_migrate 的 source/target 而异,只取绝对值)。
            if before == after:
                continue
            for k in range(6):
                diff = after[k] - before[k]
                if abs(diff) != shards[k]:
                    fails += 1
                    break
            if fails > 0:
                continue
    assert fails == 0, (label, fails)
    return len(rows)


def b2(replay_json, strategy_json, strategy_bridge, baseline_dir):
    out = {}
    # ---- oracle:决策内容全等(与离线 0/3531)----
    off = load_jsonl(os.path.join(baseline_dir, "decision_log.jsonl"))
    off_dec = {k: {d["request_id"]: d for d in off if d.get("kind") == k}
               for k in ("prefill", "decode", "completion")}
    rep = load_jsonl(replay_json)
    content_mismatch = 0
    for d in rep:
        if d["request_id"] not in off_dec[d["kind"]]:
            content_mismatch += 1
            continue
        od = off_dec[d["kind"]][d["request_id"]]
        if od.get("decision") != d.get("decision"):
            content_mismatch += 1
    out["oracle_content_mismatch"] = content_mismatch
    assert content_mismatch == 0, out
    # ---- 交叉材料核对(oracle 侧:replay 决策 == 离线决策,决策侧与离线
    # kv_cache_events.csv 同源核对;strategy 为 real-online,决策本可合法
    # 不同——实例分配差异 913 已归因,不参与 oracle 交叉核对)----
    rep_dec = {k: {d["request_id"]: d for d in rep if d.get("kind") == k}
               for k in ("prefill", "decode", "completion")}
    stra = load_jsonl(strategy_json)
    stra_dec = {k: {d["request_id"]: d for d in stra if d.get("kind") == k}
                for k in ("prefill", "decode", "completion")}
    off_rows = list(csv.DictReader(open(os.path.join(baseline_dir, "kv_cache_events.csv"))))
    off_by_key = {(r["phase"], r["event_type"], r["trigger_request_id"]): r
                  for r in off_rows}
    resp = load_bridge_responses(strategy_bridge)
    on_actions = []
    for b in resp:
        on_actions.extend(b.get("kv_actions", []))
    on_actions.sort(key=lambda a: a["event_index"])
    # history 动作:replay 预填决策 vs 离线 csv(同源 writer 交叉核对)
    hist_mismatch = 0
    hist_counter = Counter()
    for rid, d in rep_dec["prefill"].items():
        act = d["decision"]["history_action"]
        off_act = off_by_key[("history", "no_history" if act == "NO_HISTORY"
                              else ("noc_migrate" if act == "NOC_MIGRATE"
                                    else "recompute"), rid)]
        # 存在性已隐含;统计与决策侧一致的 action 计数
        hist_counter[act] += 1
        if act == "NOC_MIGRATE":
            if int(off_act["source_instance_index"]) != \
                    d["decision"]["history_source_instance_index"]:
                hist_mismatch += 1
    out["history_action_counts"] = dict(hist_counter)
    assert dict(hist_counter) == {"NO_HISTORY": 112, "NOC_MIGRATE": 302,
                                  "RECOMPUTE": 763}, hist_counter
    assert hist_mismatch == 0, out
    # prefill_decode 迁移字节:离线 noc_migrate total_bytes == replay decode
    # 决策(shards 求和 == total_bytes)
    migrate_mismatch = 0
    for rid, d in rep_dec["decode"].items():
        t = d["decision"].get("prefill_decode_transfer")
        off_r = off_by_key[("prefill_decode", "noc_migrate", rid)]
        shard_sum = sum(s["bytes"] for s in t["shards"]) \
            if t and t["shards"] and isinstance(t["shards"][0], dict) \
            else (sum(t["shards"]) if t else 0)
        if t is None or int(off_r["total_bytes"]) != t["total_bytes"] \
                or shard_sum != t["total_bytes"]:
            migrate_mismatch += 1
    out["prefill_decode_bytes_mismatch"] = migrate_mismatch
    assert migrate_mismatch == 0, out
    # eviction(oracle 交叉核对,决策侧 vs 离线 csv):851 = completion_evictions
    # 831 + decode_target_evictions 20,全部 reason=restore_1m_reserve;per
    # (request, phase) 的 shard_bytes 多重集与 csv evict_delete 行逐条一致
    off_evict = [r for r in off_rows if r["event_type"] == "evict_delete"]
    assert len(off_evict) == 851
    assert all(r["reason"] == "restore_1m_reserve" for r in off_evict)
    evict_csv_by_key = {}
    for r in off_evict:
        evict_csv_by_key.setdefault((r["phase"], r["trigger_request_id"]), [])\
            .append(json.loads(r["shard_bytes"]))
    evict_mismatch = 0
    dec_evict_entries = 0
    for rid, d in rep_dec["prefill"].items():
        for e in d["decision"]["decode_target_evictions"]:
            dec_evict_entries += 1
            rows = evict_csv_by_key.get(("decode", e["trigger_request_id"]), [])
            if not any(e["shard_bytes"] == s for s in rows):
                evict_mismatch += 1
    for rid, d in rep_dec["completion"].items():
        for e in d["decision"]["completion_evictions"]:
            dec_evict_entries += 1
            rows = evict_csv_by_key.get(("completion", e["trigger_request_id"]), [])
            if not any(e["shard_bytes"] == s for s in rows):
                evict_mismatch += 1
    assert dec_evict_entries == 851, (dec_evict_entries,)
    out["evict_mismatch"] = evict_mismatch
    out["evict_entries"] = dec_evict_entries
    assert evict_mismatch == 0, out
    # completion:retain_complete 1177 条;kv_state_after_completion EVICTED 826
    # / RESIDENT 351;EVICTED 的 retain 字节 == 末条同 tick 自驱逐 evict 字节
    # (session_22_request_10 双 evict 行,retain == 末条;4 个 RESIDENT 请求
    # 带 completion eviction 条目属语义内例外——eviction 存在而状态仍 RESIDENT,
    # 不参与 retain==evict 断言)
    off_retain = [r for r in off_rows if r["event_type"] == "retain_complete"]
    assert len(off_retain) == 1177
    ev_compl_by_req = {}
    for r in off_rows:
        if r["event_type"] == "evict_delete" and r["phase"] == "completion":
            ev_compl_by_req.setdefault(r["trigger_request_id"], [])\
                .append(json.loads(r["shard_bytes"]))
    retain_by_req = {r["trigger_request_id"]: int(r["total_bytes"]) for r in off_retain}
    comp_mismatch = 0
    evicted = resident = 0
    for rid, d in rep_dec["completion"].items():
        st = d["decision"]["kv_state_after_completion"]
        if st == "EVICTED":
            evicted += 1
            rows = ev_compl_by_req.get(rid, [])
            if not rows or retain_by_req[rid] != sum(rows[-1]):
                comp_mismatch += 1
        else:
            resident += 1
    out["completion_mismatch"] = comp_mismatch
    out["completion_state_counts"] = {"EVICTED": evicted, "RESIDENT": resident}
    assert comp_mismatch == 0 and (evicted, resident) == (826, 351), out
    # KV 账本不变量:离线 0/7932、strategy 0/5935
    off_n = _kv_ledger_invariants(off_rows, "offline")
    on_n = _kv_ledger_invariants(on_actions, "strategy")
    out["ledger_offline_rows"] = off_n
    out["ledger_strategy_rows"] = on_n
    assert off_n == 7932 and on_n == 5935, out
    # ---- real-online(strategy)逐决策不变量 C1/C2/C3 ----
    reqs = load_bridge_requests(strategy_bridge)
    arrival_order = {}
    tick_arrival_counter = {}
    for b in reqs:
        for a in b["arrivals"]:
            t = b["tick"]
            o = tick_arrival_counter.get(t, 0)
            tick_arrival_counter[t] = o + 1
            arrival_order[a["request_id"]] = (t, o)
    assert len(arrival_order) == 1177
    sys.path.insert(0, os.path.join(WL, ".."))  # llama2_7b_inference(只读导入)
    from wsc_llm_scheduler import (  # noqa: E402  (READ-ONLY, import only)
        WscLlmInstanceSpec, build_instances, build_static_pd_mapping)
    specs = tuple(WscLlmInstanceSpec(name=g.name, pg_name=g.pg_name,
                                     ranks=g.ranks, phase_role=g.phase_role)
                  for g in _load_config().inference_groups)
    topology = build_instances(config.hardware, specs, require_equal_size=True)
    static_mapping = build_static_pd_mapping(topology, alpha=1.0)
    prefill_instances = tuple(i for i in range(len(topology.instances))
                              if topology.instances[i].phase_role == "prefill")
    # C1 静态路由
    c1 = 0
    for rid, d in stra_dec["prefill"].items():
        p = d["decision"]["prefill_instance_index"]
        sr = stra_dec["decode"][rid]["decision"]["static_route"]
        expect = static_mapping.route_for_prefill(p)
        if (sr["decode_instance_index"] != expect.decode_instance_index
                or tuple(sr["path"]) != tuple(expect.path)
                or sr["hop_count"] != expect.hop_count
                or tuple(sr["shared_edges"]) != tuple(expect.shared_edges)):
            c1 += 1
    out["C1_static_route_mismatch"] = c1
    assert c1 == 0, out
    # C2 prefill argmin/assignment_key(到达 tick 快照重建 qp 深度)
    migrate_events = [a for a in on_actions if a["phase"] == "prefill_decode"
                      and a["event_type"] == "noc_migrate"]
    assert len(migrate_events) == 1177
    admit_by_request = {}
    for a in migrate_events:
        admit_by_request[a["trigger_request_id"]] = (
            a["planner_time_ns"], a["event_index"], a["target_instance_index"])
    drain_list = sorted(((a["planner_time_ns"], a["trigger_request_id"])
                         for a in migrate_events), key=lambda kv: kv[0])
    qp = [0] * len(topology.instances)
    drain_i = 0
    c2 = 0
    for rid, (t, o) in sorted(arrival_order.items(), key=lambda kv: (kv[1][0], kv[1][1])):
        while drain_i < len(drain_list) and drain_list[drain_i][0] <= t:
            qp[stra_dec["prefill"][drain_list[drain_i][1]]["decision"]
               ["prefill_instance_index"]] -= 1
            drain_i += 1
        chosen = stra_dec["prefill"][rid]["decision"]["prefill_instance_index"]
        recorded_key = tuple(stra_dec["prefill"][rid]["decision"]
                             ["prefill_assignment_key"])
        depth_vec = {i: qp[i] for i in prefill_instances}
        expect_chosen = min(prefill_instances, key=lambda i: (depth_vec[i], i))
        expect_key = (depth_vec[expect_chosen], expect_chosen)
        if chosen != expect_chosen or recorded_key != expect_key:
            c2 += 1
        qp[chosen] += 1
    out["C2_prefill_argmin_mismatch"] = c2
    assert c2 == 0, out
    # C3 decode 准入深度(admits(≤T) - completions(≤T),同 tick completion 批先)
    completion_ticks = {}
    for rid, d in stra_dec["completion"].items():
        completion_ticks.setdefault(d["tick"], []).append(rid)
    active = {}
    admit_list = sorted(admit_by_request.items(), key=lambda kv: (kv[1][0], kv[1][1]))
    admit_i = 0
    c3 = 0
    ticks = sorted(set([a[1][0] for a in admit_list]) | set(completion_ticks.keys()))
    for t in ticks:
        for rid in completion_ticks.get(t, []):
            di = stra_dec["decode"][rid]["decision"]["decode_instance_index"]
            active.get(di, set()).discard(rid)
        while admit_i < len(admit_list) and admit_list[admit_i][1][0] == t:
            rid, (at, ei, target) = admit_list[admit_i]
            recorded = stra_dec["decode"][rid]["decision"] \
                ["decode_queue_depth_before_enqueue"]
            got = len(active.get(target, set()))
            if recorded != got:
                c3 += 1
            active.setdefault(target, set()).add(rid)
            admit_i += 1
    out["C3_decode_admission_mismatch"] = c3
    assert c3 == 0, out
    out["verdict"] = "PASS"
    return out


# ----------------------------------------------------------------- B3 图结构 --


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


def b3(replay_bridge, baseline_dir, label="replay"):
    """per-(rank,request) 位置比较;within-request 依赖差异全部归入类别④。"""
    resp = load_bridge_responses(replay_bridge)
    nodes_by_rank = {}
    edges_by_rank = {}
    for b in resp:
        for n in b["nodes"]:
            nodes_by_rank.setdefault(n["rank"], []).append(n)
        for e in b["parent_edges"]:
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
    cross_rank = set()
    cross_req = set()
    req_count = 0
    missing_online = 0
    for rank in range(_load_config().npus_count):
        offline = _read_offline_et(
            os.path.join(baseline_dir, "llama2_7b_wsc_llm_inference.%d.et" % rank))
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
                    elif "history_recompute" in nm:
                        ckind = "history_recompute"
                    elif "current_prefill" in nm:
                        ckind = "current_prefill"
                    elif "_timer_gate" in nm:
                        ckind = "timer_gate"
                    elif "control" in nm:
                        ckind = "control"
                    elif "tp_ready" in nm:
                        ckind = "tp_ready"
                    elif "decode_request" in nm:
                        ckind = "decode"
                    elif "end_barrier" in nm:
                        ckind = "end_barrier"
                    else:
                        ckind = nm[:40]
                    cross_kind[ckind] += 1
                    cross_rank.add(rank)
                    cross_req.add(k)
    out = {
        "label": label,
        "responses_consumed": len(resp),
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
        "cross_affected_ranks": len(cross_rank),
        "cross_affected_requests": len(cross_req),
        "missing_ranks": missing_online,
    }
    # 登记口径(plan B 最终确定,两轮决定性验证见 phase2_status.md):
    # count/name/type/attr = 0;within 1495 全部 turn>0 transfer3000 recv
    # 节点,归入类别④;cross 差异 13165 为跨 request 依赖链差异(离线保留
    # 物理跨 request 链、replay 按 LUT 时钟语义清除 previous_id 串行化,
    # 仅保留同 session interval gate / control 显式边)——归入类别①③,
    # 属已登记差异,不 gate(与 /tmp/b3_compare.py 历史口径一致:cross 为
    # 信息性计数,RESULT 只 gate count/name/type/attr/within)。
    # 阶段 4 §7.3 修订(归因:同 tick 按冻结 queue order 稳定排序):
    # 13171 -> 13165。replay 的 13 个同 tick 多到达组中 10 组到达序 !=
    # 队列序;阶段 4 起同 tick 按 queue_index 冻结序消费,在线链与离线
    # LUT 序(离线同 tick 批序 == 队列序)一致,跨 request 依赖差异减少
    # 6(离线方向收敛);strategy 零同 tick 多到达组,决策序列逐字节不变。
    assert out["count_diffs"] == out["name_diffs"] == out["type_diffs"] == \
        out["attr_diffs"] == 0, out
    assert out["within_request_dep_diffs"] == 1495, out
    assert dict(within_kind) == {"transfer3000": 1495}, out
    assert out["within_turn_gt0"] == 1495, out
    assert out["cross_request_dep_diffs"] == 13165, out
    assert out["cross_affected_ranks"] == 54 and out["cross_affected_requests"] == 1174, out
    assert out["missing_ranks"] == 0, out
    out["verdict"] = ("PASS (count/name/type/attr 0; within 1495 all category 4; "
                      "cross 13165 registered categories 1/3, phase-4 queue-order 修订)")
    return out


# ----------------------------------------------------------------- B4 执行 --


def b4(replay_json, baseline_dir):
    """同图前提不成立,exact 不适用;操作性标准 = 差异全部可归因并记录。
    可计算的不变量:replay 完成序 vs 离线 LUT 完成序(无跨 tick 逆序)、
    无提前决策(完成不早于同 request 的 decode)。"""
    off = load_jsonl(os.path.join(baseline_dir, "decision_log.jsonl"))
    off_compl = {d["request_id"]: d for d in off if d.get("kind") == "completion"}
    off_decode = {d["request_id"]: d for d in off if d.get("kind") == "decode"}
    rep = load_jsonl(replay_json)
    rep_compl = {d["request_id"]: d for d in rep if d.get("kind") == "completion"}
    rep_seq = {d["request_id"]: d["seq"] for d in rep if d.get("kind") == "completion"}
    # 完成序一致性:离线 (tick, seq) 全序 vs 在线交付序——跨 tick 逆序 = 0
    # (硬门);同 tick 组内顺序自由(③ tick-end/deferred 顺序合同,组内逆序
    # 对数为信息性计数)。
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
    # 提前决策:在线 completion 不得早于同 request 的在线 decode 决策交付
    # (阶段序在交付空间同样成立;tick 空间已在 B1 阶段序断言)。
    rep_decode_seq = {d["request_id"]: d["seq"] for d in rep if d.get("kind") == "decode"}
    early = sum(1 for rid, seq in rep_seq.items()
                if seq <= rep_decode_seq[rid])
    inversions = cross_inv
    out = {
        "completion_cross_tick_inversions": inversions,
        "completion_within_tick_inversion_pairs": within_inv,
        "completion_before_decode": early,
        "attribution": {
            "1_replay_rig_LUT_clock": "concurrent calibrated COMP chains; comm-0 "
            "(RECV side) 1ns instant completion; prefill chain clearing; calibrated "
            "runtimes shift completion boundaries earlier (completion deltas in "
            "[-17,-1])",
            "2_cross_request_serialization_edges": "replay emits 0 cross-request "
            "previous_id edges by design (LUT authority = decision_log); offline "
            ".et keeps physical chains; intentional difference (contract 7, "
            "attribution category 1)",
            "3_tick_end_deferred_ordering": "tick-end/deferred delivery contract; "
            "decode deltas in [-12,+1] with 1 positive (session_41_request_0)",
            "4_decode_group_none_restore": "5abaf23 None-restore: decode-group "
            "rank block-end = None; 1495 within-request transfer3000 recv edges "
            "missing own-request parents; two decisive rounds proved restoring "
            "them desyncs replay (delivery 55 / tick 5057359000) -- necessary "
            "runtime fix, not an over-broad regression",
        },
    }
    assert inversions == 0 and early == 0, out
    out["verdict"] = "PASS (operational standard: all differences attributable and recorded)"
    return out


# -------------------------------------------------- B-LEGACY(阶段 7 §10.6) --


def b_legacy(legacy_run, legacy_baseline):
    """legacy 第二变体(real-online)对照层。

    对照材料:在线 legacy 运行目录 results/(online_decision_log.jsonl +
    kv_event_payload_legacy.json)vs 离线 legacy baseline(调用方按方案文档
    §3 步骤 0-1 物化的目录,含 decision_log.jsonl + manifest.json)。
    合同⑦ Tier B real-online 口径:完成 tick 由真实物理链决定,不要求与
    离线决策序列 exact——硬门只验**策略结构性不变量**(与完成顺序无关):
      1. 行数:离线/在线决策行 3531 = 3×1177(prefill+decode+completion;
         离线 69481 iteration 行在线无对应物,不参与);
      2. 覆盖:1177 request,每 kind 恰好 1 行,request 集一致;
      3. 阶段序(delivery seq 空间):prefill < decode < completion;
      4. 决策字段形状:每 kind 恰好离线 writer 的字段集(防漂移);
      5. legacy 常量字段:effective_prefill_tokens=0 / history_action=
         NO_HISTORY / UNTRACKED / 空 eviction / transfer=None(legacy
         策略永不驱逐,离线/在线同值);
      6. history_transfer_bytes 逐 request 全等(同一冻结输入事实推导,
         顺序无关);
      7. terminal_kv_release_at_completion 逐 request 全等(仅由输入
         事实决定);
      8. 链内一致性(在线自身):turn>0 的 history_source_instance_index
         == 前序 turn 的 decode_instance_index;turn-0 为 None;
      9. C1 静态路由(在线自身):decode 目标 == route_for_prefill;
     10. allocator 终值:在线 kv_event_payload_legacy == 离线 manifest
         kv_management.final_remaining_capacity_bytes(同一
         WscRelevantKvAllocator、全 terminal 释放后 = 初始容量)。
    顺序敏感字段(prefill_instance_index / prefill_assignment_key /
    decode_instance_index / static_route / decode_queue_depth /
    estimated_arrival_ns)为**信息性差异报告**(不 gate,差异可解释性
    证据),tick 差异分布同报。
    """
    kinds = ("prefill", "decode", "completion")
    out = {}
    off = load_jsonl(os.path.join(legacy_baseline, "decision_log.jsonl"))
    off_dec = [d for d in off if d.get("kind") in kinds]
    on = load_jsonl(os.path.join(
        legacy_run, "results", "online_decision_log.jsonl"))
    assert len(off_dec) == 3531, len(off_dec)
    assert len(on) == 3531, len(on)
    off_by = {k: {d["request_id"]: d for d in off_dec if d["kind"] == k}
              for k in kinds}
    on_by = {k: {d["request_id"]: d for d in on if d["kind"] == k}
             for k in kinds}
    for k in kinds:
        assert len(off_by[k]) == len(on_by[k]) == 1177, (
            k, len(off_by[k]), len(on_by[k]))
        assert set(off_by[k]) == set(on_by[k]), k
    out["requests"] = 1177

    # 3. 阶段序(在线 delivery seq 空间)。
    stage_violations = 0
    for rid in on_by["prefill"]:
        if not (on_by["prefill"][rid]["seq"] < on_by["decode"][rid]["seq"]
                < on_by["completion"][rid]["seq"]):
            stage_violations += 1
    out["stage_order_violations"] = stage_violations
    assert stage_violations == 0, out

    # 4. 决策字段形状。
    off_keys = {k: sorted(off_by[k][next(iter(off_by[k]))]["decision"])
                for k in kinds}
    shape_fail = 0
    for k in kinds:
        for rid, d in on_by[k].items():
            if sorted(d["decision"]) != off_keys[k]:
                shape_fail += 1
    out["decision_shape_fail"] = shape_fail
    assert shape_fail == 0, out

    # 5. legacy 常量字段。
    const_fail = 0
    for rid, d in on_by["prefill"].items():
        dec = d["decision"]
        if (dec["effective_prefill_tokens"] != 0
                or dec["history_action"] != "NO_HISTORY"
                or dec["history_cache_state_before"] != "UNTRACKED"
                or dec["history_recompute_tokens"] != 0
                or dec["admission_evictions"] != []
                or dec["decode_target_evictions"] != []):
            const_fail += 1
    for rid, d in on_by["decode"].items():
        if d["decision"]["prefill_decode_transfer"] is not None:
            const_fail += 1
    for rid, d in on_by["completion"].items():
        dec = d["decision"]
        if (dec["kv_state_after_completion"] != "UNTRACKED"
                or dec["kv_instance_after_completion"] is not None
                or dec["completion_evictions"] != []):
            const_fail += 1
    out["legacy_const_field_fail"] = const_fail
    assert const_fail == 0, out

    # 6. history_transfer_bytes 确定性(离线/在线同一冻结事实)。
    hist_bytes_fail = 0
    for rid, d in on_by["prefill"].items():
        if (d["decision"]["history_transfer_bytes"]
                != off_by["prefill"][rid]["decision"]["history_transfer_bytes"]):
            hist_bytes_fail += 1
    out["history_bytes_fail"] = hist_bytes_fail
    assert hist_bytes_fail == 0, out

    # 7. terminal 确定性。
    terminal_fail = 0
    for rid, d in on_by["completion"].items():
        if (d["decision"]["terminal_kv_release_at_completion"]
                != off_by["completion"][rid]["decision"]
                ["terminal_kv_release_at_completion"]):
            terminal_fail += 1
    out["terminal_fail"] = terminal_fail
    assert terminal_fail == 0, out

    # 8. 链内一致性(在线自身;request_id = <session>_request_<turn>)。
    decode_by_turn = {}
    for rid, d in on_by["decode"].items():
        session, turn = rid.rsplit("_request_", 1)
        decode_by_turn[(session, int(turn))] = \
            d["decision"]["decode_instance_index"]
    chain_fail = 0
    for rid, d in on_by["prefill"].items():
        session, turn = rid.rsplit("_request_", 1)
        turn = int(turn)
        source = d["decision"]["history_source_instance_index"]
        if turn == 0:
            if source is not None:
                chain_fail += 1
        elif source != decode_by_turn.get((session, turn - 1)):
            chain_fail += 1
    out["history_chain_fail"] = chain_fail
    assert chain_fail == 0, out

    # 9. C1 静态路由(在线自身一致,与 B2 C1 同构)。
    sys.path.insert(0, os.path.join(WL, ".."))
    from wsc_llm_scheduler import (  # noqa: E402  (READ-ONLY, import only)
        WscLlmInstanceSpec, build_instances, build_static_pd_mapping)
    specs = tuple(WscLlmInstanceSpec(name=g.name, pg_name=g.pg_name,
                                     ranks=g.ranks, phase_role=g.phase_role)
                  for g in _load_config().inference_groups)
    topology = build_instances(config.hardware, specs, require_equal_size=True)
    static_mapping = build_static_pd_mapping(topology, alpha=1.0)
    c1 = 0
    for rid, d in on_by["prefill"].items():
        p = d["decision"]["prefill_instance_index"]
        sr = on_by["decode"][rid]["decision"]["static_route"]
        expect = static_mapping.route_for_prefill(p)
        if (sr["decode_instance_index"] != expect.decode_instance_index
                or tuple(sr["path"]) != tuple(expect.path)
                or sr["hop_count"] != expect.hop_count
                or tuple(sr["shared_edges"]) != tuple(expect.shared_edges)
                or sr["prefill_instance_index"] != expect.prefill_instance_index):
            c1 += 1
    out["C1_static_route_mismatch"] = c1
    assert c1 == 0, out

    # 10. allocator 终值:在线 payload == 离线 manifest 同口径终值。
    payload = json.load(open(os.path.join(
        legacy_run, "results", "kv_event_payload_legacy.json")))
    manifest = json.load(open(os.path.join(legacy_baseline, "manifest.json")))
    off_final = manifest["kv_management"]["final_remaining_capacity_bytes"]
    assert payload["policy"] == "wsc_relevant_pd_static_decode_domain", payload
    assert len(payload["final_remaining_capacity_bytes"]) == 9, payload
    assert payload["final_remaining_capacity_bytes"] == off_final, (
        payload["final_remaining_capacity_bytes"], off_final)
    out["allocator_final_equal"] = True

    # ---- 信息性差异报告(不 gate;合同⑦差异可解释性证据)----
    content_exact = sum(
        1 for rid in on_by["prefill"]
        if all(on_by[k][rid]["decision"] == off_by[k][rid]["decision"]
               for k in kinds))
    out["content_exact_requests"] = content_exact
    field_diffs = Counter()
    for k in kinds:
        for rid in on_by[k]:
            od = off_by[k][rid]["decision"]
            nd = on_by[k][rid]["decision"]
            for field in set(od) | set(nd):
                if od.get(field) != nd.get(field):
                    field_diffs[k + "." + field] += 1
    out["field_diffs"] = dict(field_diffs)
    tick_deltas = {}
    for k in kinds:
        deltas = [on_by[k][rid]["tick"] - off_by[k][rid]["tick"]
                  for rid in on_by[k]]
        tick_deltas[k] = {
            "exact": sum(1 for v in deltas if v == 0),
            "min": min(deltas), "max": max(deltas),
            "mean": int(sum(deltas) / len(deltas)),
        }
    out["tick_deltas"] = tick_deltas
    out["verdict"] = "PASS"
    return out


# ---------------------------------------------------------------------- main --


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay-bridge", default="/tmp/wscllm_run_phase2_replay/bridge")
    ap.add_argument("--strategy-bridge", default="/tmp/wscllm_run_phase2_strategy/bridge")
    ap.add_argument("--baseline", default=None,
                    help="离线基线目录(调用方按方案文档 §3 步骤 0-1 物化后显式传入;"
                    "缺失 fail-closed)")
    ap.add_argument("--replay-json", default=None)
    ap.add_argument("--strategy-json", default=None)
    # 阶段 7 §10.6:legacy 在线运行目录(给定则运行 B-LEGACY 层)。
    ap.add_argument("--legacy-run", default=None,
                    help="legacy 在线运行目录(results/ 含 online_decision_log."
                    "jsonl 与 kv_event_payload_legacy.json)")
    ap.add_argument("--legacy-baseline", default=None,
                    help="legacy 离线基线目录(调用方按方案文档 §3 步骤 0-1 物化后"
                    "显式传入;缺失 fail-closed)")
    ap.add_argument("--config", default=None,
                    help="物化后的 trace 配置(拓扑事实来源;B2/B3/B-LEGACY 需要;"
                    "缺失 fail-closed)")
    args = ap.parse_args()
    # request-neutral:仓库不预置物化输入,相关参数缺失直接 fail-closed。
    global _CONFIG_PATH
    _CONFIG_PATH = args.config
    if args.legacy_run:
        missing = ("--legacy-baseline" if not args.legacy_baseline else
                   "--config" if not args.config else None)
    else:
        missing = ("--baseline" if not args.baseline else
                   "--config" if not args.config else None)
    if missing:
        print("missing %s: request-neutral 仓库不预置物化输入;请按方案文档 "
              "wscllm仓库改造详细执行方案.md §3 步骤 0-1 物化后显式传入" % missing,
              file=sys.stderr)
        sys.exit(1)
    replay_json = args.replay_json or os.path.join(args.replay_bridge,
                                                   "online_decision_log.jsonl")
    strategy_json = args.strategy_json or os.path.join(args.strategy_bridge,
                                                       "online_decision_log.jsonl")
    results = {}
    if args.legacy_run:
        # 阶段 7 §10.6:legacy 分支独立统计(两分支独立性红线)。--legacy-run
        # 模式只跑 B-LEGACY 层——B0-B4 是 session_lru/replay 分支的对照
        # (默认桥目录为阶段 2 旧 run,对 legacy 运行无意义);legacy 分支的
        # 对照物是离线 legacy baseline(--legacy-baseline)。
        results["B-LEGACY"] = b_legacy(args.legacy_run, args.legacy_baseline)
    else:
        results["B0"] = b0(replay_json, strategy_json, args.baseline)
        results["B1"] = b1(replay_json, args.baseline)
        results["B2"] = b2(replay_json, strategy_json, args.strategy_bridge, args.baseline)
        results["B3"] = b3(args.replay_bridge, args.baseline)
        results["B4"] = b4(replay_json, args.baseline)
    print("=" * 78)
    print("TIER B COMPARISON SUMMARY")
    print("=" * 78)
    for layer, r in results.items():
        print("[%s] verdict: %s" % (layer, r["verdict"]))
    if "B1" in results:
        print("--- B1 detail ---")
        for kind in ("prefill", "decode", "completion"):
            d = results["B1"][kind]
            print("  %-10s total=%d exact=%d negative=%d positive=%d range=%s" % (
                kind, d["total"], d["exact"], d["negative"], d["positive"],
                d["delta_range"]))
        print("--- B2 detail ---")
        b2r = results["B2"]
        print("  oracle content mismatch:", b2r["oracle_content_mismatch"])
        print("  history action counts:", b2r["history_action_counts"])
        print("  prefill_decode bytes mismatch:", b2r["prefill_decode_bytes_mismatch"])
        print("  evict mismatch:", b2r["evict_mismatch"])
        print("  completion mismatch:", b2r["completion_mismatch"],
              "state counts:", b2r["completion_state_counts"])
        print("  ledger rows offline/strategy: %d/%d" % (
            b2r["ledger_offline_rows"], b2r["ledger_strategy_rows"]))
        print("  C1/C2/C3 mismatch:", b2r["C1_static_route_mismatch"],
              b2r["C2_prefill_argmin_mismatch"], b2r["C3_decode_admission_mismatch"])
        print("--- B3 detail ---")
        b3r = results["B3"]
        print("  request groups compared:", b3r["request_groups_compared"])
        print("  count/name/type/attr diffs: %d/%d/%d/%d" % (
            b3r["count_diffs"], b3r["name_diffs"], b3r["type_diffs"], b3r["attr_diffs"]))
        print("  within-request dep diffs:", b3r["within_request_dep_diffs"],
              "by kind:", b3r["within_by_kind"], "(all turn>0)")
        print("  cross-request dep diffs:", b3r["cross_request_dep_diffs"])
        print("--- B4 detail ---")
        b4r = results["B4"]
        print("  completion cross-tick inversions:", b4r["completion_cross_tick_inversions"])
        print("  completion before decode:", b4r["completion_before_decode"])
        for k, v in b4r["attribution"].items():
            print("  [%s] %s" % (k, v))
    if "B-LEGACY" in results:
        print("--- B-LEGACY detail (phase 7 §10.6) ---")
        leg = results["B-LEGACY"]
        print("  requests:", leg["requests"],
              "stage_order_violations:", leg["stage_order_violations"],
              "decision_shape_fail:", leg["decision_shape_fail"])
        print("  legacy_const_field_fail:", leg["legacy_const_field_fail"],
              "history_bytes_fail:", leg["history_bytes_fail"],
              "terminal_fail:", leg["terminal_fail"],
              "history_chain_fail:", leg["history_chain_fail"],
              "C1_static_route_mismatch:", leg["C1_static_route_mismatch"])
        print("  allocator_final_equal:", leg["allocator_final_equal"])
        print("  content_exact_requests (decision 全等,含顺序敏感字段):",
              leg["content_exact_requests"], "/", leg["requests"])
        print("  field_diffs (离线 vs 在线,顺序敏感字段为信息性差异):")
        for field, count in sorted(leg["field_diffs"].items()):
            print("    %-52s %d" % (field, count))
        for k, d in leg["tick_deltas"].items():
            print("  tick delta %-10s exact=%d min=%d max=%d mean=%d" % (
                k, d["exact"], d["min"], d["max"], d["mean"]))
    print("=" * 78)
    print("ALL LAYERS PASS")


if __name__ == "__main__":
    main()
