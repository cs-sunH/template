#!/usr/bin/env python3
"""WP4 canonical cache 事件适配器（逐仓适配；五仓文件逐字节相同，语义差异
全部收在 REPO_VARIANTS 表；标准库实现）。

输入 = run_dir/results/*.jsonl（online_decision_log.jsonl 为主；per-request
manifest 提供 turn_index 等上下文）。只做输出层映射，native 日志保持原样，
不反向参与调度（主规格第一原则 b）。

输出：
  * canonical ``cache_events.csv``：action_id,request_id,start_ns,end_ns,
    bytes,source,target,cause。start_ns=决策/记录发射 tick；end_ns=NA
    （物理完成 tick 属 full 档 memory_anchor，由 restore_decomposition.py
    配对，本层不臆造）。
  * 每请求 ``kv_hit_state``（full/partial/miss/no_history/not_supported；
    no_history 不计入 miss）+ 证据列（native 字段=值）。
  * 命中率双分母（主规格 §3.2）：(full+partial)/有历史请求数 与
    (full+partial)/全部请求数；no_history 单列。

--reconcile：canonical bytes/count 按 family 与 native 日志逐项对账
（以 native 为准），不一致 exit!=0。

逐仓字段映射证据（基线产物 60s_summary 实测 + 代码行号）：
  astra-sim-face   prefill.decision.history_action ∈ {NO_HISTORY, LOCAL_HIT,
                   NOC_MIGRATE, RECOMPUTE}；RECOMPUTE 的
                   history_recompute_tokens==history_tokens_before
                   （83/83 全量重算 → miss，无 partial 语义）。
  astra-sim-wscllm 同 face（RECOMPUTE 802/802 全量重算）；decode 决策另有
                   static_route（hopbytes.py 用）。
  astra-sim-sh_1.0 prefill.decision.history_transfer ∈ null | {kind:
                   noc_migrate|local_hit|remote_load, shards[{
                   bytes,noc_path,source_rank,target_rank}...],
                   total_bytes,...}；完成/逐出为 remote_store 等传输对象。
                   local_hit 无数据 shard（generate_face_trace.py:541），
                   total_bytes 为名义规模——不产 canonical 事件。
  astra-sim-sh_2.0 决策只落 history_transfer_bytes /
                   prefill_decode_transfer_bytes / *_eviction_count 聚合值；
                   准入时点的 history_location_before 在运行态存在但未序列化
                   （sh20_online_scheduler.py:1104-1108 解包、:1159-1167
                   决策字典未含）→ turn>0 的 kv_hit_state=not_supported。
  astra-sim-sh_3.0 prefill.decision.prefill_affinity_reason ∈
                   {first_request_non_edge, first_request_edge_fallback,
                   resident_local_hbm, resident_prefix_layers,
                   remote_edge_load_balance}（sh30_online_scheduler.py:
                   1150-1221 三分支）；PARTIAL_HBM_REMOTE 驻留 → partial。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slo_common import (  # noqa: E402
    DECISION_LOG_RELPATH, NA, detect_repo_variant, emit_json, fail,
    iter_jsonl, load_request_manifest, manifest_requests, open_output,
    run_main, write_csv,
)

CACHE_EVENT_COLUMNS = ("action_id", "request_id", "start_ns", "end_ns",
                       "bytes", "source", "target", "cause")
KV_HIT_STATE_COLUMNS = ("request_id", "turn_index", "kv_hit_state", "evidence")

FAMILY_HISTORY = "history_transfer"
FAMILY_PD = "prefill_decode_transfer"
FAMILY_EVICTION = "eviction"


def _turn_index_map(manifest: dict) -> dict[str, int]:
    result = {}
    for entry in manifest_requests(manifest):
        request_id = entry.get("request_id")
        turn = entry.get("turn_index")
        if isinstance(request_id, str) and isinstance(turn, int):
            result[request_id] = turn
    return result


# ---------------------------------------------------------------------------
# kv_hit_state 映射（逐仓）
# ---------------------------------------------------------------------------

def _hit_state_face_wscllm(decision: dict, turn: Optional[int],
                           request_id: str) -> tuple[str, str]:
    action = decision.get("history_action")
    state_before = decision.get("history_cache_state_before")
    evidence = f"history_action={action};history_cache_state_before={state_before}"
    mapping = {
        "NO_HISTORY": "no_history",
        "LOCAL_HIT": "full",
        "NOC_MIGRATE": "full",
        "RECOMPUTE": "miss",
    }
    if action not in mapping:
        return "not_supported", evidence
    return mapping[action], evidence


def _hit_state_sh10(decision: dict, turn: Optional[int],
                    request_id: str) -> tuple[str, str]:
    transfer = decision.get("history_transfer")
    if transfer is None:
        if turn is not None and turn > 0:
            # 基线中 null 恰为 turn-0（116/116）；turn>0 且无记录属异常，
            # 不猜测。
            return ("not_supported",
                    f"history_transfer=null;turn_index={turn}")
        return "no_history", "history_transfer=null;turn_index=0"
    kind = transfer.get("kind")
    reason = transfer.get("reason")
    evidence = f"history_transfer.kind={kind};reason={reason}"
    mapping = {
        "local_hit": "full",
        "noc_migrate": "full",
        # remote_load：历史整体驻留远存并恢复（非重算）→ full 命中，
        # 恢复成本进入 restore 分解。
        "remote_load": "full",
    }
    if kind not in mapping:
        return "not_supported", evidence
    return mapping[kind], evidence


def _hit_state_sh20(decision: dict, turn: Optional[int],
                    request_id: str) -> tuple[str, str]:
    if turn is not None and turn == 0:
        return "no_history", "turn_index=0（首请求无历史）"
    if turn is None:
        return ("not_supported",
                "manifest 缺 turn_index，且 sh_2.0 决策未落准入时点历史位置")
    return ("not_supported",
            "sh_2.0 decision log 未序列化 admission 时点 history_location"
            "_before（代码证据 sh20_online_scheduler.py:1104-1167）——"
            "不得由聚合 bytes 猜测 full/miss")


def _hit_state_sh30(decision: dict, turn: Optional[int],
                    request_id: str) -> tuple[str, str]:
    reason = decision.get("prefill_affinity_reason")
    source = decision.get("history_source_instance_index")
    discarded = decision.get("history_tokens_discarded")
    evidence = (f"prefill_affinity_reason={reason};"
                f"history_source_instance_index={source};"
                f"history_tokens_discarded={discarded}")
    mapping = {
        "first_request_non_edge": "no_history",
        "first_request_edge_fallback": "no_history",
        "resident_local_hbm": "full",
        "resident_prefix_layers": "partial",
        "remote_edge_load_balance": "full",
    }
    if reason not in mapping:
        return "not_supported", evidence
    return mapping[reason], evidence


# ---------------------------------------------------------------------------
# canonical 事件提取（逐仓）
# ---------------------------------------------------------------------------

def _event(request_id: str, tick: int, nbytes: Optional[int], source: Any,
           target: Any, cause: str) -> dict:
    return {
        "request_id": request_id,
        "start_ns": tick,
        "end_ns": None,
        "bytes": int(nbytes) if nbytes is not None else None,
        "source": NA if source is None else source,
        "target": NA if target is None else target,
        "cause": cause,
    }


def _eviction_bytes_face(entry: dict) -> Optional[int]:
    shard_bytes = entry.get("shard_bytes")
    if isinstance(shard_bytes, list) and shard_bytes:
        return int(sum(shard_bytes))
    return None


def extract_events_face_wscllm(record: dict) -> list[dict]:
    """face/wscllm：history_action + prefill_decode_transfer + 逐出。"""
    events = []
    kind = record.get("kind")
    decision = record.get("decision") or {}
    request_id = record.get("request_id")
    tick = record.get("tick")
    if not isinstance(tick, int):
        fail(f"decision log 行缺整数 tick（request={request_id!r}）")
    if kind == "prefill":
        nbytes = decision.get("history_transfer_bytes")
        if isinstance(nbytes, int) and nbytes > 0:
            events.append(_event(
                request_id, tick, nbytes,
                decision.get("history_source_instance_index"),
                decision.get("prefill_instance_index"),
                f"{FAMILY_HISTORY}:history_action="
                f"{decision.get('history_action')}"))
        for field in ("admission_evictions", "decode_target_evictions"):
            for entry in decision.get(field) or []:
                nbytes = _eviction_bytes_face(entry)
                if nbytes:
                    events.append(_event(
                        entry.get("trigger_request_id") or request_id,
                        entry.get("time_ns", tick), nbytes,
                        entry.get("victim_instance_index"), None,
                        f"{FAMILY_EVICTION}:{entry.get('reason')}"))
    elif kind == "decode":
        transfer = decision.get("prefill_decode_transfer")
        if isinstance(transfer, dict):
            nbytes = transfer.get("total_bytes")
            if isinstance(nbytes, int) and nbytes > 0:
                events.append(_event(
                    transfer.get("trigger_request_id") or request_id, tick,
                    nbytes, transfer.get("source_instance_index"),
                    transfer.get("target_instance_index"),
                    f"{FAMILY_PD}:{transfer.get('reason')}"))
    elif kind == "completion":
        for entry in decision.get("completion_evictions") or []:
            nbytes = _eviction_bytes_face(entry)
            if nbytes:
                events.append(_event(
                    entry.get("trigger_request_id") or request_id,
                    entry.get("time_ns", tick), nbytes,
                    entry.get("victim_instance_index"), None,
                    f"{FAMILY_EVICTION}:{entry.get('reason')}"))
    return events


def _transfer_object_events(transfer: dict, default_request: str, tick: int,
                            family: str) -> list[dict]:
    """S1 传输对象 → canonical 事件（bytes 取 total_bytes，缺失则 Σshards）。"""
    if not isinstance(transfer, dict):
        return []
    if transfer.get("kind") == "local_hit":
        return []
    nbytes = transfer.get("total_bytes")
    if not isinstance(nbytes, int):
        shards = transfer.get("shards")
        if isinstance(shards, list):
            nbytes = sum(int(s.get("bytes", 0)) for s in shards
                         if isinstance(s, dict))
    if not nbytes:
        return []
    return [_event(
        transfer.get("trigger_request_id") or default_request, tick, nbytes,
        transfer.get("source_instance_index"),
        transfer.get("target_instance_index"),
        f"{family}:{transfer.get('kind')}:{transfer.get('reason')}")]


def extract_events_sh10(record: dict) -> list[dict]:
    events = []
    kind = record.get("kind")
    decision = record.get("decision") or {}
    request_id = record.get("request_id")
    tick = record.get("tick")
    if not isinstance(tick, int):
        fail(f"decision log 行缺整数 tick（request={request_id!r}）")
    if kind == "prefill":
        events.extend(_transfer_object_events(
            decision.get("history_transfer"), request_id, tick,
            FAMILY_HISTORY))
        for field in ("history_evictions", "prefill_evictions"):
            for entry in decision.get(field) or []:
                events.extend(_transfer_object_events(
                    entry, request_id, tick, FAMILY_EVICTION))
    elif kind == "decode":
        events.extend(_transfer_object_events(
            decision.get("prefill_decode_transfer"), request_id, tick,
            FAMILY_PD))
        for entry in decision.get("decode_evictions") or []:
            events.extend(_transfer_object_events(
                entry, request_id, tick, FAMILY_EVICTION))
    elif kind == "completion":
        for entry in decision.get("completion_evictions") or []:
            events.extend(_transfer_object_events(
                entry, request_id, tick, FAMILY_EVICTION))
    return events


def extract_events_sh20(record: dict) -> list[dict]:
    """S2：只有聚合 bytes，无 shard/路由/逐出字节（计数不产事件）。"""
    events = []
    kind = record.get("kind")
    decision = record.get("decision") or {}
    request_id = record.get("request_id")
    tick = record.get("tick")
    if not isinstance(tick, int):
        fail(f"decision log 行缺整数 tick（request={request_id!r}）")
    if kind == "prefill":
        nbytes = decision.get("history_transfer_bytes")
        if isinstance(nbytes, int) and nbytes > 0:
            events.append(_event(
                request_id, tick, nbytes, NA, NA,
                f"{FAMILY_HISTORY}:aggregate_bytes_only"))
    elif kind == "decode":
        nbytes = decision.get("prefill_decode_transfer_bytes")
        if isinstance(nbytes, int) and nbytes > 0:
            events.append(_event(
                request_id, tick, nbytes, NA, NA,
                f"{FAMILY_PD}:aggregate_bytes_only"))
    return events


def extract_events_sh30(record: dict) -> list[dict]:
    events = []
    kind = record.get("kind")
    decision = record.get("decision") or {}
    request_id = record.get("request_id")
    tick = record.get("tick")
    if not isinstance(tick, int):
        fail(f"decision log 行缺整数 tick（request={request_id!r}）")
    if kind == "prefill":
        nbytes = decision.get("history_transfer_bytes")
        if isinstance(nbytes, int) and nbytes > 0:
            events.append(_event(
                request_id, tick, nbytes,
                decision.get("history_source_instance_index"),
                decision.get("prefill_instance_index"),
                f"{FAMILY_HISTORY}:reason="
                f"{decision.get('prefill_affinity_reason')}"))
    elif kind == "completion":
        for entry in decision.get("completion_evictions") or []:
            nbytes = entry.get("total_bytes") if isinstance(entry, dict) else None
            if isinstance(nbytes, int) and nbytes > 0:
                layer_note = ""
                if isinstance(entry, dict) and "layer_start" in entry:
                    layer_note = (f";layers={entry.get('layer_start')}-"
                                  f"{entry.get('layer_end')}")
                events.append(_event(
                    entry.get("trigger_request_id") or request_id, tick,
                    nbytes, entry.get("source_instance_index"),
                    entry.get("target_instance_index"),
                    f"{FAMILY_EVICTION}:{entry.get('kind')}:"
                    f"{entry.get('reason')}{layer_note}"))
    return events


REPO_VARIANTS: dict[str, dict[str, Any]] = {
    "astra-sim-face": {
        "hit_state": _hit_state_face_wscllm,
        "extract_events": extract_events_face_wscllm,
        "shard_sum_check": True,
        "partial_semantics": False,
        "notes": "history_action 四值（60s_summary: NO_HISTORY 116/"
                 "LOCAL_HIT 176/NOC_MIGRATE 1079/RECOMPUTE 83）；"
                 "RECOMPUTE 全量重算（83/83）→ 无 partial 语义",
    },
    "astra-sim-wscllm": {
        "hit_state": _hit_state_face_wscllm,
        "extract_events": extract_events_face_wscllm,
        "shard_sum_check": True,
        "partial_semantics": False,
        "notes": "同 face（RECOMPUTE 802/802 全量重算）；PD 迁移 100% "
                 "NOC_MIGRATE；decode 另有 static_route（hopbytes.py）",
    },
    "astra-sim-sh_1.0": {
        "hit_state": _hit_state_sh10,
        "extract_events": extract_events_sh10,
        "shard_sum_check": True,
        "partial_semantics": False,
        "notes": "history_transfer.kind ∈ {noc_migrate 1135, local_hit 174, "
                 "remote_load 29}、null 116（恰为 turn-0）；shards 带 "
                 "noc_path（hopbytes.py 用）",
    },
    "astra-sim-sh_2.0": {
        "hit_state": _hit_state_sh20,
        "extract_events": extract_events_sh20,
        "shard_sum_check": False,
        "partial_semantics": True,
        "notes": "决策仅聚合 bytes（history_transfer_bytes>0 1086）；"
                 "admission 时点 history_location_before 未序列化"
                 "（sh20_online_scheduler.py:1104-1167）→ kv_hit_state="
                 "not_supported（turn0 除外）；TODO_S2：需 native 侧补落"
                 "位置字段后重评",
    },
    "astra-sim-sh_3.0": {
        "hit_state": _hit_state_sh30,
        "extract_events": extract_events_sh30,
        "shard_sum_check": False,
        "partial_semantics": True,
        "notes": "affinity_reason 分布（60s_summary）：first_request_"
                 "non_edge 116/resident_local_hbm 1239/remote_edge_load_"
                 "balance 80/resident_prefix_layers 19（→partial）；代码 "
                 "sh30_online_scheduler.py:1150-1221",
    },
}


def family_of(cause: str) -> str:
    return cause.split(":", 1)[0]


def cmd_adapter(args: argparse.Namespace) -> int:
    repo_variant = detect_repo_variant(args.run_dir, args.repo_variant)
    variant = REPO_VARIANTS.get(repo_variant)
    if variant is None:
        fail(f"未登记的 repo_variant：{repo_variant}（REPO_VARIANTS 需扩表，"
             f"禁止猜测映射）")
    manifest = load_request_manifest(args.run_dir, args.request_manifest)
    turns = _turn_index_map(manifest)

    events: list[dict] = []
    hit_states: dict[str, tuple[str, str]] = {}
    shard_sum_violations = 0
    log_path = args.run_dir / DECISION_LOG_RELPATH
    for record in iter_jsonl(log_path):
        decision = record.get("decision") or {}
        request_id = record.get("request_id")
        if record.get("kind") == "prefill" and isinstance(request_id, str):
            state, evidence = variant["hit_state"](
                decision, turns.get(request_id), request_id)
            hit_states[request_id] = (state, evidence)
        extracted = variant["extract_events"](record)
        events.extend(extracted)
        if variant["shard_sum_check"]:
            # native 不变量复检：Σshards == total_bytes（以 native 为准）。
            for holder in (decision.get("prefill_decode_transfer"),
                           decision.get("history_transfer")):
                if isinstance(holder, dict):
                    total = holder.get("total_bytes")
                    shards = holder.get("shards")
                    if (isinstance(total, int) and isinstance(shards, list)
                            and shards):
                        shard_total = sum(int(s.get("bytes", 0))
                                          for s in shards
                                          if isinstance(s, dict))
                        if shard_total != total:
                            shard_sum_violations += 1
            for field in ("completion_evictions", "history_evictions",
                          "prefill_evictions", "decode_evictions"):
                for holder in decision.get(field) or []:
                    if isinstance(holder, dict):
                        total = holder.get("total_bytes")
                        shards = holder.get("shards")
                        if isinstance(total, int) and isinstance(shards, list):
                            shard_total = sum(int(s.get("bytes", 0))
                                              for s in shards
                                              if isinstance(s, dict))
                            if shard_total != total:
                                shard_sum_violations += 1

    manifest_ids = [str(e.get("request_id")) for e in
                    manifest_requests(manifest) if e.get("request_id")]
    state_rows = []
    n_no_prefill_decision = 0
    for request_id in manifest_ids:
        if request_id in hit_states:
            state, evidence = hit_states[request_id]
        else:
            n_no_prefill_decision += 1
            state, evidence = ("not_supported",
                               "decision log 缺该请求 prefill 决策")
        state_rows.append((request_id, turns.get(request_id, NA), state,
                           evidence))
    counts = {"full": 0, "partial": 0, "miss": 0, "no_history": 0,
              "not_supported": 0}
    for _, _, state, _ in state_rows:
        counts[state] = counts.get(state, 0) + 1
    n_all = len(state_rows)
    with_history = n_all - counts["no_history"]
    hit_n = counts["full"] + counts["partial"]
    summary = {
        "command": "kv_cache_adapter",
        "repo_variant": repo_variant,
        "notes": variant["notes"],
        "partial_semantics_native": variant["partial_semantics"],
        "n_requests": n_all,
        "kv_hit_state_counts": counts,
        "hit_rate_over_requests_with_history": (
            hit_n / with_history) if with_history else None,
        "hit_rate_over_all_requests": (hit_n / n_all) if n_all else None,
        "no_history_not_counted_as_miss": True,
        "n_events": len(events),
        "n_requests_without_prefill_decision": n_no_prefill_decision,
        "shard_sum_violations": shard_sum_violations,
    }

    out_path = None
    if args.output == "-":
        if args.reconcile:
            fail("--reconcile 需要可重读的 canonical CSV：'-o -'（stdout）"
                 "无法回读对账；用 -o 指定文件路径，或省略 -o"
                 "（缺省写 run_dir/cache_events.csv）")
    elif args.output != "":
        out_path = Path(args.output)
    stream, close = open_output(args.output, "cache_events.csv", args.run_dir)
    try:
        write_csv(
            stream, CACHE_EVENT_COLUMNS,
            ((f"a{index:06d}", event["request_id"], event["start_ns"],
              NA, event["bytes"], event["source"], event["target"],
              event["cause"])
             for index, event in enumerate(events, start=1)))
    finally:
        if close:
            stream.close()
    hs_stream, hs_close = open_output(args.hit_states, "kv_hit_states.csv",
                                      args.run_dir)
    try:
        write_csv(hs_stream, KV_HIT_STATE_COLUMNS, state_rows)
    finally:
        if hs_close:
            hs_stream.close()
    emit_json(sys.stderr, summary)
    if args.json:
        jstream, jclose = open_output(args.json, "kv_cache_adapter.json",
                                      args.run_dir)
        try:
            emit_json(jstream, summary)
        finally:
            if jclose:
                jstream.close()

    if args.reconcile:
        return reconcile(run_dir=args.run_dir, repo_variant=repo_variant,
                         variant=variant, events=events, out_path=out_path,
                         summary=summary)
    return 0


def reconcile(run_dir: Path, repo_variant: str, variant: dict,
              events: list[dict], out_path: Optional[Path],
              summary: dict) -> int:
    """canonical bytes/count vs native 逐项对账（以 native 为准）。

    native 侧：独立从 decision log 重放聚合（按 family）；canonical 侧：
    重新解析已写出的 cache_events.csv 聚合。
    """
    import csv as _csv

    native: dict[str, dict[str, int]] = {}
    for record in iter_jsonl(run_dir / DECISION_LOG_RELPATH):
        decision = record.get("decision") or {}
        request_id = record.get("request_id")
        kind = record.get("kind")
        tick = record.get("tick")
        buckets = []
        if kind == "prefill":
            nbytes = decision.get("history_transfer_bytes")
            transfer = decision.get("history_transfer")
            if isinstance(transfer, dict):
                if transfer.get("kind") == "local_hit":
                    nbytes = 0  # 名义 KV 规模，无物理搬运
                else:
                    total = transfer.get("total_bytes")
                    if isinstance(total, int):
                        nbytes = total
            if isinstance(nbytes, int) and nbytes > 0:
                buckets.append((FAMILY_HISTORY, nbytes))
            for field in ("admission_evictions", "decode_target_evictions",
                          "history_evictions", "prefill_evictions"):
                for entry in decision.get(field) or []:
                    if isinstance(entry, dict) and entry.get(
                            "kind") == "local_hit":
                        continue
                    nbytes = _eviction_bytes_face(entry)
                    if isinstance(entry, dict) and not nbytes:
                        total = entry.get("total_bytes")
                        shards = entry.get("shards")
                        if not isinstance(total, int) and isinstance(
                                shards, list) and shards:
                            total = sum(int(s.get("bytes", 0))
                                        for s in shards
                                        if isinstance(s, dict))
                        nbytes = total if isinstance(total, int) else None
                    if nbytes:
                        buckets.append((FAMILY_EVICTION, nbytes))
        elif kind == "decode":
            transfer = decision.get("prefill_decode_transfer")
            nbytes = decision.get("prefill_decode_transfer_bytes")
            if isinstance(transfer, dict):
                if transfer.get("kind") == "local_hit":
                    nbytes = 0  # 名义 KV 规模，无物理搬运
                else:
                    total = transfer.get("total_bytes")
                    if isinstance(total, int):
                        nbytes = total
            if isinstance(nbytes, int) and nbytes > 0:
                buckets.append((FAMILY_PD, nbytes))
            for entry in decision.get("decode_evictions") or []:
                if isinstance(entry, dict) and entry.get(
                        "kind") != "local_hit":
                    total = entry.get("total_bytes")
                    shards = entry.get("shards")
                    if not isinstance(total, int) and isinstance(
                            shards, list) and shards:
                        total = sum(int(s.get("bytes", 0)) for s in shards
                                    if isinstance(s, dict))
                    if isinstance(total, int) and total > 0:
                        buckets.append((FAMILY_EVICTION, total))
        elif kind == "completion":
            for entry in decision.get("completion_evictions") or []:
                nbytes = None
                if isinstance(entry, dict) and entry.get(
                        "kind") == "local_hit":
                    continue
                if isinstance(entry, dict):
                    nbytes = _eviction_bytes_face(entry)
                    total = entry.get("total_bytes")
                    shards = entry.get("shards")
                    if not nbytes:
                        if isinstance(total, int):
                            nbytes = total
                        elif isinstance(shards, list):
                            nbytes = sum(int(s.get("bytes", 0))
                                         for s in shards
                                         if isinstance(s, dict))
                if nbytes:
                    buckets.append((FAMILY_EVICTION, nbytes))
        for family, nbytes in buckets:
            slot = native.setdefault(family, {"count": 0, "bytes": 0})
            slot["count"] += 1
            slot["bytes"] += int(nbytes)

    csv_path = out_path if out_path is not None else (
        run_dir / "cache_events.csv")
    canonical: dict[str, dict[str, int]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = _csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CACHE_EVENT_COLUMNS:
            fail(f"canonical CSV 列序不符：{csv_path}")
        for row in reader:
            family = family_of(row["cause"])
            slot = canonical.setdefault(family, {"count": 0, "bytes": 0})
            slot["count"] += 1
            slot["bytes"] += int(row["bytes"])

    mismatches = []
    families = sorted(set(native) | set(canonical))
    report = []
    for family in families:
        n_slot = native.get(family, {"count": 0, "bytes": 0})
        c_slot = canonical.get(family, {"count": 0, "bytes": 0})
        ok = (n_slot["count"] == c_slot["count"]
              and n_slot["bytes"] == c_slot["bytes"])
        report.append({
            "family": family,
            "native_count": n_slot["count"],
            "canonical_count": c_slot["count"],
            "native_bytes": n_slot["bytes"],
            "canonical_bytes": c_slot["bytes"],
            "ok": ok,
        })
        if not ok:
            mismatches.append(family)
    payload = {
        "command": "kv_cache_adapter --reconcile",
        "repo_variant": repo_variant,
        "basis": "native（decision log）为准确侧；canonical 差异即映射错误",
        "families": report,
        "shard_sum_violations": summary.get("shard_sum_violations"),
        "ok": not mismatches,
    }
    emit_json(sys.stderr, payload)
    if mismatches:
        fail(f"对账不一致（family: {mismatches}）——canonical 事件表与 "
             f"native 日志逐项对账失败，以 native 为准修正映射")
    if summary.get("shard_sum_violations"):
        fail(f"native 不变量复检失败：Σshards != total_bytes 共 "
             f"{summary['shard_sum_violations']} 条")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="kv_cache_adapter.py",
        description="WP4 canonical cache 事件适配器（native→canonical 只映射，"
                    "native 日志保持原样）")
    parser.add_argument("run_dir", type=Path,
                        help="运行目录（含 results/online_decision_log.jsonl）")
    parser.add_argument("-o", "--output", default="",
                        help="cache_events.csv 输出（'-'=stdout；缺省写 "
                             "run_dir/cache_events.csv）")
    parser.add_argument("--hit-states", default="",
                        help="kv_hit_states.csv 输出（同上；缺省写 "
                             "run_dir/kv_hit_states.csv）")
    parser.add_argument("--json", default="",
                        help="汇总 JSON 输出路径（可选）")
    parser.add_argument("--reconcile", action="store_true",
                        help="canonical bytes/count 与 native 日志逐项对账"
                             "（不一致 exit!=0，以 native 为准）")
    parser.add_argument("--repo-variant", default=None,
                        help="显式指定 repo_variant（默认读 cpp.log init 行）")
    parser.add_argument("--request-manifest", type=Path, default=None,
                        help="per-request manifest（默认 run_dir/"
                             "metrics_manifest.json 或 cpp.log init 行）")
    args = parser.parse_args()
    return int(cmd_adapter(args) or 0)


if __name__ == "__main__":
    sys.exit(run_main(main))
