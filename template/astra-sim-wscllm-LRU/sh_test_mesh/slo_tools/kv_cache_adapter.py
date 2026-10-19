#!/usr/bin/env python3
"""WP4 canonical cache 事件适配器（逐仓适配；语义差异
全部收在 REPO_VARIANTS 表；标准库实现）。

同步基线注记（2026-09-26）：本文件与 face/wscllm 家族同名脚本已按仓分叉，
不再跨仓逐字节相同；语义差异仍以 REPO_VARIANTS 表为准。

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
  astra-sim-face   双级（-LRU 方案 A）：新产物 prefill.decision 恒携带
                   history_location_before（三态 local_hbm/
                   partial_hbm_remote/remote_memory → full/partial/full，
                   照 sh_2.0）+ history_transfers 逐段对象 + 契约逐出行
                   （history_evictions/prefill_evictions/decode_evictions，
                   B3 双序列化的 legacy admission/decode_target/
                   completion_evictions 不再消费——同值同源防双计）；
                   旧产物回退 history_action 四值 {NO_HISTORY, LOCAL_HIT,
                   NOC_MIGRATE, RECOMPUTE}。
  astra-sim-wscllm 同 face 双级（旧基线 RECOMPUTE 802/802 全量重算；
                   completion 契约行沿用 completion_evictions 字段名，
                   legacy 8 字段行按形状分流）；decode 决策另有
                   static_route（hopbytes.py 用）。
  astra-sim-sh_1.0 prefill.decision.history_transfer ∈ null | {kind:
                   noc_migrate|local_hit|remote_load, shards[{
                   bytes,noc_path,source_rank,target_rank}...],
                   total_bytes,...}；完成/逐出为 remote_store 等传输对象。
                   local_hit 无数据 shard（sh_1.0 仓 generate_face_trace.py），
                   total_bytes 为名义规模——不产 canonical 事件。
  astra-sim-sh_2.0 决策只落 history_transfer_bytes /
                   prefill_decode_transfer_bytes / *_eviction_count 聚合值;
                   WP9-线5（2026-08-26）起 admission 决策附加序列化
                   history_location_before（三态：local_hbm /
                   partial_hbm_remote / remote_memory）+
                   history_resident_prefix_layers → full/partial/full
                   （remote 整体恢复非重算;无 recompute,miss 不出现）。
  astra-sim-sh_3.0 prefill.decision.prefill_affinity_reason ∈
                   {first_request_non_edge, first_request_edge_fallback,
                   resident_local_hbm, resident_prefix_layers,
                   remote_edge_load_balance}（sh30_online_scheduler.py 的
                   affinity 三分支）；PARTIAL_HBM_REMOTE 驻留 → partial。
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
        # （2026-10-01）bool 排除：isinstance(True, int) 为真，bool turn
        # 会与真实 turn 1 的字典键冲突。
        if (isinstance(request_id, str) and isinstance(turn, int)
                and not isinstance(turn, bool)):
            result[request_id] = turn
    return result


# ---------------------------------------------------------------------------
# kv_hit_state 映射（逐仓）
# ---------------------------------------------------------------------------

def _hit_state_face_wscllm(decision: dict, turn: Optional[int],
                           request_id: str) -> tuple[str, str]:
    """face/wscllm 双级（-LRU 方案 A）：新产物优先 history_location_before
    三态映射（照 sh_2.0 仓），字段缺失回退 legacy history_action
    四值表（旧产物兼容）。

    session 级 Tiered-LRU 改造后新运行的值域为 {local_hbm, remote_memory}
    → 恒 full（partial 仅旧产物可达，映射表保留只为旧日志解析）；
    history_action 值域已扩（REMOTE_RESTORE 等）且 RECOMPUTE 消失——
    这些行恒携带 history_location_before（NO_HISTORY 恒 None，走回退级
    恰得 no_history），不需要也不应扩 action 表。五值域不扩（REMOTE→
    full、PARTIAL→partial；full_local/full_remote 由 evidence 列区分），
    命中率分子 hit_n=full+partial 公式不动（partial 分支新运行恒空）。"""
    location = decision.get("history_location_before")
    if location is not None:
        resident = decision.get("history_resident_prefix_layers")
        evidence = (
            f"history_location_before={location};"
            f"resident_prefix_layers={resident};"
            f"instance_index="
            f"{decision.get('history_location_before_instance_index')}")
        mapping = {
            "local_hbm": "full",
            "partial_hbm_remote": "partial",
            "remote_memory": "full",
        }
        if location not in mapping:
            return "not_supported", evidence
        return mapping[location], evidence
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
    if not isinstance(transfer, dict):
        # 非 dict 真值（str/int/bool 等）旧行为裸 AttributeError → 退出码
        # 1，与 fail-closed 2 的失败语义契约不符。
        fail(f"history_transfer 非对象（request={request_id!r}，实得 "
             f"{type(transfer).__name__}）——形状谎报，fail-closed")
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
    """S2：准入时点会话历史位置（WP9-线5 起 decision 序列化
    history_location_before）映射三态语义。

    local_hbm → full（全量驻留本_instance HBM）；partial_hbm_remote →
    partial（resident 前缀层在 HBM、suffix 在远存）；remote_memory →
    full（整体驻留远存并恢复，非重算，与 S1 remote_load 口径一致）。
    S2 无 recompute 语义，miss 不出现属预期（映射表保持封闭集合，
    未知位置 fail 到 not_supported 不猜测）。旧日志（字段缺失，B2
    线4 基线）保持 not_supported，可区分新旧产物。"""
    location = decision.get("history_location_before")
    if location is None:
        if turn is not None and turn == 0:
            # 首请求无历史：turn-0 无位置快照是正常态（SessionKVSnapshot
            # 仅在会话已有 KV 状态时返回）。
            return "no_history", "turn_index=0（首请求无历史）"
        return ("not_supported",
                "sh_2.0 decision log 未序列化 admission 时点 history_location"
                "_before（旧产物；新产物见 WP9-线5 序列化字段）")
    resident = decision.get("history_resident_prefix_layers")
    evidence = (
        f"history_location_before={location};"
        f"resident_prefix_layers={resident};"
        f"instance_index={decision.get('history_location_before_instance_index')}")
    mapping = {
        "local_hbm": "full",
        "partial_hbm_remote": "partial",
        "remote_memory": "full",
    }
    if location not in mapping:
        return "not_supported", evidence
    return mapping[location], evidence


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


def _hit_state_wscllm(decision: dict, turn: Optional[int],
                      request_id: str) -> tuple[str, str]:
    """wscllm 仓内 policy 收口（A.5/2026-09-05）：与 face 共用双级口径
    （-LRU 方案 A；tiered 值域下 slo_tools 只按 repo_variant 分发，
    不再按 policy 分叉）。"""
    return _hit_state_face_wscllm(decision, turn, request_id)


# ---------------------------------------------------------------------------
# canonical 事件提取（逐仓）
# ---------------------------------------------------------------------------

def _event(request_id: str, tick: int, nbytes: Optional[int], source: Any,
           target: Any, cause: str) -> dict:
    # KV-01：负 bytes（账本损坏）不得进入普通输出路径——本函数是全部
    # 逐仓提取路径的唯一事件构造点，在此统一拒绝（不等到 --reconcile
    # 才暴露不一致）。
    if nbytes is not None and int(nbytes) < 0:
        fail(f"canonical 事件 bytes 为负（{nbytes}，request={request_id!r},"
             f" cause={cause}）——账本损坏，拒绝输出")
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
    """legacy shard_bytes 列表优先；缺则回退 total_bytes 标量（-LRU 契约
    §3 的 sh 形状逐出条目是传输对象：kind/total_bytes/layer_start/layer_end，
    无 shard_bytes——不回退则被静默丢弃）。"""
    shard_bytes = entry.get("shard_bytes")
    if isinstance(shard_bytes, list) and shard_bytes:
        # 元素零校验（2026-10-01 对齐）：sum() 对字符串元素裸抛 TypeError
        # 会逃出 fail() 契约退出码；bool 元素按 1 字节计。逐元素严格非负
        # 整数（拒 bool），违规走本层 fail() 契约通道。
        for element in shard_bytes:
            if not isinstance(element, int) or isinstance(element, bool)                     or element < 0:
                fail("逐出条目 shard_bytes 元素必须是非负整数"
                     f"(got {element!r}, trigger_request_id="
                     f"{entry.get('trigger_request_id')!r})")
        return int(sum(shard_bytes))
    total = entry.get("total_bytes")
    # bool 排除（2026-10-01 对齐）：total_bytes=true 曾按 1 字节放行。
    if (isinstance(total, int) and not isinstance(total, bool)
            and total > 0):
        return total
    return None


def _row_where(record: dict) -> str:
    """决策行定位上下文（hbm_watermark 的 where 纪律同款：iter_jsonl 不
    外发物理行号，seq 即决策日志的行身份；运行时报错信息允许携带它）。"""
    return (f"seq={record.get('seq', '?')} request="
            f"{record.get('request_id')!r} kind={record.get('kind')!r}")


def extract_events_face_wscllm(record: dict,
                               state: "AdapterState" = None) -> list[dict]:
    """face/wscllm：双级（-LRU 方案 A）——history_transfers 列表在场
    （新产物恒在场，含空列表）走契约口径产逐段传输事件；契约分段逐出
    字段（history_evictions/prefill_evictions）在场消费契约逐出行，缺席
    （B2 窗口产物/旧产物）回退 legacy 聚合 bytes + shard_bytes/total_
    bytes 逐出条目。两级逐出互斥：legacy admission_evictions/decode_
    target_evictions 与契约行同值同源（B3 双序列化），同批消费会双计。

    新旧产物判别式 = 契约分段逐出字段（与 hopbytes 单键取用同式）而非
    history_transfers：后者 B2 就在场而分段逐出字段 B3 才加——按
    history_transfers 判新会把 B2 窗口产物误走契约逐出行（字段缺席 →
    legacy 逐出整类静默丢失、reconcile 两边同丢恒 ok）。

    legacy 回退的跨行去重（F1，2026-09-30）：B2 窗口产物的同一笔 #2
    预占逐出在 prefill 行（准入快照）与 decode 行（全集）各序列化一次
    ——prefill 回退消费后把去重键记入 state.legacy_eviction_keys，
    decode 回退跳过已记键（契约路径不受影响：分段字段/decode_evictions
    在场时本函数不触碰 legacy 字段，state 恒空）。state=None 时（独立
    单行调用）退化为旧行为。"""
    events = []
    kind = record.get("kind")
    decision = record.get("decision") or {}
    request_id = record.get("request_id")
    tick = record.get("tick")
    # bool 排除（2026-10-01 对齐）：isinstance(True, int) 为真，tick=true
    # 曾按 tick=1 放行。
    if isinstance(tick, bool) or not isinstance(tick, int):
        fail(f"decision log 行缺整数 tick（request={request_id!r}）")
    if kind == "prefill":
        seen_keys = (state.legacy_eviction_keys.setdefault(
            str(request_id), set()) if state is not None else None)
        transfers = decision.get("history_transfers")
        if isinstance(transfers, list):
            for entry in transfers:
                if not isinstance(entry, dict):
                    # HBM-04/INPUT-03 同口径（R5-batch3，2026-10-01）：
                    # 非 dict 成员不得静默过滤——损坏行只会缩小逐出/传输
                    # 字节账，且 canonical/native 双侧同丢会让 --reconcile
                    # 恒过零留痕。
                    fail(f"decision log history_transfers 成员必须是对象"
                         f"（{_row_where(record)}）——损坏行 fail-closed")
                events.extend(_transfer_object_events(
                    entry, request_id, tick, FAMILY_HISTORY))
        else:
            nbytes = decision.get("history_transfer_bytes")
            if isinstance(nbytes, int) and not isinstance(nbytes, bool) and nbytes > 0:
                events.append(_event(
                    request_id, tick, nbytes,
                    decision.get("history_source_instance_index"),
                    decision.get("prefill_instance_index"),
                    f"{FAMILY_HISTORY}:history_action="
                    f"{decision.get('history_action')}"))
        if "history_evictions" in decision \
                or "prefill_evictions" in decision:
            for field in ("history_evictions", "prefill_evictions"):
                for entry in decision.get(field) or []:
                    if not isinstance(entry, dict):
                        fail(f"decision log {field} 成员必须是对象"
                             f"（{_row_where(record)}）——损坏行 "
                             f"fail-closed")
                    events.extend(_transfer_object_events(
                        entry, request_id, tick, FAMILY_EVICTION))
        else:
            for field in ("admission_evictions", "decode_target_evictions"):
                for entry in decision.get(field) or []:
                    if not isinstance(entry, dict):
                        # 非 dict 成员此前在 _eviction_bytes_face 的
                        # entry.get 上裸 AttributeError 逃出 fail-closed
                        # 出口码 2——统一为结构化 fail（行上下文内嵌）。
                        fail(f"decision log {field} 成员必须是对象"
                             f"（{_row_where(record)}）——损坏行 "
                             f"fail-closed")
                    nbytes = _eviction_bytes_face(entry)
                    if nbytes:
                        events.append(_event(
                            entry.get("trigger_request_id") or request_id,
                            entry.get("time_ns", tick), nbytes,
                            entry.get("victim_instance_index"), None,
                            f"{FAMILY_EVICTION}:{entry.get('reason')}"))
                    if seen_keys is not None:
                        key = _legacy_eviction_key(entry)
                        if key is not None:
                            seen_keys.add(key)
    elif kind == "decode":
        # 逐出单键取用：契约字段 decode_evictions 在场（B3+）优先；缺席
        # （B2 窗口/旧产物）回退 legacy decode_target_evictions——decode
        # 侧逐出自 B2 起随 decode 行落盘（KVTransfer 契约行结构），不回
        # 退则 B2 逐出整类不入账；F1 去重：跳过 prefill 行准入快照已消费
        # 的 #2 条目（同键=同一次物理逐出的双序列化）。
        if "decode_evictions" in decision:
            eviction_rows = decision.get("decode_evictions")
        else:
            eviction_rows = decision.get("decode_target_evictions")
        seen_keys = (state.legacy_eviction_keys.get(str(request_id))
                     if state is not None and "decode_evictions" not in
                     decision else None)
        for entry in eviction_rows or []:
            if not isinstance(entry, dict):
                fail(f"decision log 逐出行成员必须是对象"
                     f"（{_row_where(record)}）——损坏行 fail-closed")
            if seen_keys:
                key = _legacy_eviction_key(entry)
                if key is not None and key in seen_keys:
                    continue
            events.extend(_transfer_object_events(
                entry, request_id, tick, FAMILY_EVICTION))
        transfer = decision.get("prefill_decode_transfer")
        if isinstance(transfer, dict):
            # local_hit 守卫（R2-3，2026-09-30，joint 同款修复同步）：
            # local_hit 的 total_bytes 是名义整份 KV（信息量，非物理流量
            # ——调度器 moved_bytes=0、_transfer_object_events 同排除），
            # 计入会使 canonical 的 P→D 家族字节在 stay 边界虚增整份
            # 会话 KV。现生产不可达（source==target 仅防御/单测路径），
            # 防御补齐。
            nbytes = transfer.get("total_bytes")
            if transfer.get("kind") == "local_hit":
                nbytes = 0
            if isinstance(nbytes, int) and not isinstance(nbytes, bool) and nbytes > 0:
                events.append(_event(
                    transfer.get("trigger_request_id") or request_id, tick,
                    nbytes, transfer.get("source_instance_index"),
                    transfer.get("target_instance_index"),
                    f"{FAMILY_PD}:{transfer.get('reason')}"))
        else:
            # 旧产物标量回退（与 prefill 侧 history_transfer_bytes 回退、
            # extract_events_sh20/reconcile 的 aggregate 口径同模式）。
            nbytes = decision.get("prefill_decode_transfer_bytes")
            if isinstance(nbytes, int) and not isinstance(nbytes, bool) and nbytes > 0:
                events.append(_event(
                    request_id, tick, nbytes, NA, NA,
                    f"{FAMILY_PD}:aggregate_bytes_only"))
    elif kind == "completion":
        for entry in decision.get("completion_evictions") or []:
            if not isinstance(entry, dict):
                fail(f"decision log completion_evictions 成员必须是对象"
                     f"（{_row_where(record)}）——非 dict 此前在 legacy "
                     f"回退臂裸 AttributeError 逃出 fail-closed 出口码 2")
            if entry.get("kind"):
                # -LRU 契约行（kind 在场即契约形状；legacy 8 字段行无
                # kind，走 shard_bytes/total_bytes 回退口径）。wscllm 完成
                # 路径零逐出恒空，按契约消费保持同构。
                events.extend(_transfer_object_events(
                    entry, request_id, tick, FAMILY_EVICTION))
                continue
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
    """S1 传输对象 → canonical 事件（bytes 取 total_bytes，缺失则 Σshards）。

    非 dict 首参在此静默返回空——这只服务 holder 字段缺席的旧产物形态
    （sh10/sh20/sh30 以 decision.get("history_transfer") 等可缺席 holder
    直调本函数）；face/wscllm 路径的列表成员由 extract_events_face_
    wscllm/reconcile 在调用前做 HBM-04 同款非 dict fail-closed 门，
    不得把成员损坏伪装成"字段缺席"静默吞掉（R5-batch3，2026-10-01）。"""
    if not isinstance(transfer, dict):
        return []
    if transfer.get("kind") == "local_hit":
        return []
    nbytes = transfer.get("total_bytes")
    # R3-1(2026-09-30):顶层 total_bytes 同排 bool(True 经 isinstance(int)
    # 混入会按 1 字节计)。
    if isinstance(nbytes, bool) or not isinstance(nbytes, int):
        nbytes = None
        shards = transfer.get("shards")
        if isinstance(shards, list):
            # 严格 int 门（F6，2026-09-30）：shards 在场而 bytes 类型谎报
            # （字符串数字/浮点/bool——int() 会把 "1024"/12.0/True 静默
            # 收编）不得静默按 0/截断值重放，与 hbm_watermark 的
            # "在场但类型谎报 fail-closed" 同口径。
            total = 0
            for shard in shards:
                if not isinstance(shard, dict):
                    fail(f"shard 元素非对象（{family}，request="
                         f"{default_request!r}）——形状谎报 fail-closed")
                value = shard.get("bytes")
                if isinstance(value, bool) or not isinstance(value, int) \
                        or value < 0:
                    fail(f"shard bytes 非非负整数（{family}，request="
                         f"{default_request!r}，bytes={value!r}）——"
                         f"类型谎报 fail-closed")
                total += value
            nbytes = total
    if not nbytes:
        return []
    return [_event(
        transfer.get("trigger_request_id") or default_request, tick, nbytes,
        transfer.get("source_instance_index"),
        transfer.get("target_instance_index"),
        f"{family}:{transfer.get('kind')}:{transfer.get('reason')}")]


def _legacy_eviction_key(entry: dict) -> Optional[tuple]:
    """legacy 逐出条目的跨行去重键（F1，2026-09-30）。

    B2 窗口产物（分段逐出字段与 decode_evictions 均缺席）里，同一笔
    #2 预占逐出既随 prefill 行的 decode_target_evictions（准入快照）
    又随 decode 行的 decode_target_evictions（#2+#4+#5 全集）双序列化
    ——_kv_transfer_rows 行结构不携带 time_ns，同一 KVTransfer 的两次
    序列化以 (session_id, reason, total_bytes) 三元组恒等（同一会话在
    一个请求持有期内不可能以同 reason 同字节被逐出两次：逐出即 REMOTE，
    再逐须经他人恢复 + 完成，reason/时点必变）。"""
    if not isinstance(entry, dict):
        return None
    return (str(entry.get("session_id")), str(entry.get("reason")),
            entry.get("total_bytes"))


def _strict_shard_sum(shards: list, context: str) -> int:
    """Σshards 的严格 int 门（F6，2026-09-30）：与 _transfer_object_events
    同口径——shard 元素非对象或 bytes 非 non-bool int 即 fail-closed，
    不静默按 0/截断值求和。"""
    total = 0
    for shard in shards:
        if not isinstance(shard, dict):
            fail(f"{context}: shard 元素非对象——形状谎报 fail-closed")
        value = shard.get("bytes")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            fail(f"{context}: shard bytes 非非负整数（bytes={value!r}）——"
                 f"类型谎报 fail-closed")
        total += value
    return total


def extract_events_sh10(record: dict, state: "AdapterState" = None) -> list[dict]:
    del state  # 家族统一签名（wscllm 侧 F1 去重态；本变体无跨行状态）
    events = []
    kind = record.get("kind")
    decision = record.get("decision") or {}
    request_id = record.get("request_id")
    tick = record.get("tick")
    # bool 排除（2026-10-01 对齐）：isinstance(True, int) 为真，tick=true
    # 曾按 tick=1 放行。
    if isinstance(tick, bool) or not isinstance(tick, int):
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


def extract_events_sh20(record: dict, state: "AdapterState" = None) -> list[dict]:
    del state  # 家族统一签名（wscllm 侧 F1 去重态；本变体无跨行状态）
    """S2：B3-6（2026-08-27）起新产物逐条序列化 *_evictions 与
    history_transfers——字段在场时按 S1 同构逐条产事件（victim 归因
    trigger 请求行，layer 区间进 cause 证据由 reason/kind 承载）；
    旧产物只有聚合 bytes（history_transfer_bytes /
    prefill_decode_transfer_bytes，计数不产事件）→ aggregate_bytes_only
    回退，两代产物可区分。"""
    events = []
    kind = record.get("kind")
    decision = record.get("decision") or {}
    request_id = record.get("request_id")
    tick = record.get("tick")
    # bool 排除（2026-10-01 对齐）：isinstance(True, int) 为真，tick=true
    # 曾按 tick=1 放行。
    if isinstance(tick, bool) or not isinstance(tick, int):
        fail(f"decision log 行缺整数 tick（request={request_id!r}）")
    if kind == "prefill":
        transfers = decision.get("history_transfers")
        if isinstance(transfers, list):
            for entry in transfers:
                events.extend(_transfer_object_events(
                    entry, request_id, tick, FAMILY_HISTORY))
        else:
            nbytes = decision.get("history_transfer_bytes")
            if isinstance(nbytes, int) and not isinstance(nbytes, bool) and nbytes > 0:
                events.append(_event(
                    request_id, tick, nbytes, NA, NA,
                    f"{FAMILY_HISTORY}:aggregate_bytes_only"))
        for field in ("history_evictions", "prefill_evictions"):
            for entry in decision.get(field) or []:
                events.extend(_transfer_object_events(
                    entry, request_id, tick, FAMILY_EVICTION))
    elif kind == "decode":
        decode_list = decision.get("decode_evictions")
        if isinstance(decode_list, list):
            for entry in decode_list:
                events.extend(_transfer_object_events(
                    entry, request_id, tick, FAMILY_EVICTION))
        nbytes = decision.get("prefill_decode_transfer_bytes")
        if isinstance(nbytes, int) and not isinstance(nbytes, bool) and nbytes > 0:
            events.append(_event(
                request_id, tick, nbytes, NA, NA,
                f"{FAMILY_PD}:aggregate_bytes_only"))
    elif kind == "completion":
        completion_list = decision.get("completion_evictions")
        if isinstance(completion_list, list):
            for entry in completion_list:
                events.extend(_transfer_object_events(
                    entry, request_id, tick, FAMILY_EVICTION))
    return events


def extract_events_sh30(record: dict, state: "AdapterState" = None) -> list[dict]:
    del state  # 家族统一签名（wscllm 侧 F1 去重态；本变体无跨行状态）
    events = []
    kind = record.get("kind")
    decision = record.get("decision") or {}
    request_id = record.get("request_id")
    tick = record.get("tick")
    # bool 排除（2026-10-01 对齐）：isinstance(True, int) 为真，tick=true
    # 曾按 tick=1 放行。
    if isinstance(tick, bool) or not isinstance(tick, int):
        fail(f"decision log 行缺整数 tick（request={request_id!r}）")
    if kind == "prefill":
        nbytes = decision.get("history_transfer_bytes")
        if isinstance(nbytes, int) and not isinstance(nbytes, bool) and nbytes > 0:
            events.append(_event(
                request_id, tick, nbytes,
                decision.get("history_source_instance_index"),
                decision.get("prefill_instance_index"),
                f"{FAMILY_HISTORY}:reason="
                f"{decision.get('prefill_affinity_reason')}"))
    elif kind == "completion":
        for entry in decision.get("completion_evictions") or []:
            nbytes = entry.get("total_bytes") if isinstance(entry, dict) else None
            if isinstance(nbytes, int) and not isinstance(nbytes, bool) and nbytes > 0:
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


def extract_events_wscllm(record: dict,
                           state: "AdapterState" = None) -> list[dict]:
    """wscllm 仓内 policy 收口（A.5/2026-09-05）：与 face 共用双级提取
    口径（-LRU 方案 A；tiered 值域下 slo_tools 只按 repo_variant 分发，
    不再按 policy 分叉）。"""
    return extract_events_face_wscllm(record, state)


REPO_VARIANTS: dict[str, dict[str, Any]] = {
    "astra-sim-face": {
        "hit_state": _hit_state_face_wscllm,
        "extract_events": extract_events_face_wscllm,
        "shard_sum_check": True,
        "partial_semantics": True,
        "notes": "双级（-LRU 方案 A）：新产物 history_location_before 三态"
                 "（local_hbm=full/partial_hbm_remote=partial/"
                 "remote_memory=full）+ history_transfers 逐段/契约逐出行；"
                 "旧产物 history_action 四值（60s_summary: NO_HISTORY 116/"
                 "LOCAL_HIT 176/NOC_MIGRATE 1079/RECOMPUTE 83；RECOMPUTE "
                 "全量重算 → 无 partial 语义）",
    },
    "astra-sim-wscllm": {
        "hit_state": _hit_state_wscllm,
        "extract_events": extract_events_wscllm,
        "shard_sum_check": True,
        "partial_semantics": True,
        "notes": "同 face 双级（-LRU 新产物契约字段优先，旧产物回退 "
                 "history_action 四值；旧基线 RECOMPUTE 802/802 全量重算）；"
                 "PD 迁移 100% NOC_MIGRATE；decode 另有 static_route"
                 "（hopbytes.py）",
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
                 "WP9-线5（2026-08-26）起 admission 决策序列化 "
                 "history_location_before{,_instance_index} 与 "
                 "history_resident_prefix_layers → kv_hit_state 三态映射"
                 "（local_hbm=full/partial_hbm_remote=partial/"
                 "remote_memory=full；无 recompute，miss 不出现属预期）；"
                 "旧产物（字段缺失）保持 not_supported",
    },
    "astra-sim-sh_3.0": {
        "hit_state": _hit_state_sh30,
        "extract_events": extract_events_sh30,
        "shard_sum_check": False,
        "partial_semantics": True,
        "notes": "affinity_reason 分布（60s_summary）：first_request_"
                 "non_edge 116/resident_local_hbm 1239/remote_edge_load_"
                 "balance 80/resident_prefix_layers 19（→partial）；代码 "
                 "sh_3.0 仓 sh30_online_scheduler.py 的 affinity 三分支",
    },
}


def family_of(cause: str) -> str:
    return cause.split(":", 1)[0]


class AdapterState:
    """单遍累积态（A4：driver 单遍 decision log 复用；CLI 路径同构）。

    events 为紧凑 6 元组列表（2026-08-30 阶段2加固 §3.4a），元素下标：
    0=request_id / 1=start_ns / 2=bytes / 3=source / 4=target / 5=cause
    （end_ns 恒 NA 不缓存，CSV 写出层补 NA；None bytes 由 write_csv 统一
    转 NA）。原每事件 ~350B dict 降为 ~120B 元组。仅两处消费：emit 的
    len()（n_events）与 canonical CSV 写出（按下标取值）。注意：不得改成
    "边消费边写最终 CSV"——driver 是单遍 sink 扇出架构，kv sink 可能中途
    死亡且不调 emit，边写会在死 sink 场景留下半截产物，改变失败语义。
    """

    __slots__ = ("events", "hit_states", "shard_sum_violations", "n_records",
                 "legacy_eviction_keys", "seen_kinds")

    def __init__(self) -> None:
        self.events: list[tuple] = []
        self.hit_states: dict[str, tuple[str, str]] = {}
        self.shard_sum_violations = 0
        self.n_records = 0
        # F1（2026-09-30）：legacy 回退路径的跨行逐出去重键
        # （request_id → {(session_id, reason, total_bytes), ...}）。
        self.legacy_eviction_keys: dict[str, set] = {}
        # F5（2026-09-30）：三种决策 kind 的判重表（与 hbm_watermark 的
        # seen_kinds 同口径——重复 decode/completion 行会把逐出/PD 事件
        # 双计进 cache_events.csv）。
        self.seen_kinds: dict[str, set] = {}


def adapter_prepare(repo_variant: str, manifest: dict
                    ) -> tuple[dict, dict[str, int], AdapterState]:
    variant = REPO_VARIANTS.get(repo_variant)
    if variant is None:
        fail(f"未登记的 repo_variant：{repo_variant}（REPO_VARIANTS 需扩表，"
             f"禁止猜测映射）")
    turns = _turn_index_map(manifest)
    return variant, turns, AdapterState()


def adapter_consume(record: dict, variant: dict, turns: dict[str, int],
                    state: AdapterState) -> None:
    """单条决策记录的适配器处理（与独立 CLI 的循环体逐语句等价）。"""
    decision = record.get("decision") or {}
    request_id = record.get("request_id")
    state.n_records += 1
    kind = record.get("kind")
    # F5（2026-09-30）：判重扩展到全部三种决策 kind——生产 writer 每请求
    # 每种 kind 恰写一行；重复 decode/completion 行会双计逐出/PD 事件
    # （standalone --reconcile 因 native 侧同样双计而恒过，防线盲）。
    if kind in ("prefill", "decode", "completion") and isinstance(
            request_id, str):
        seen = state.seen_kinds.setdefault(request_id, set())
        if kind in seen:
            fail(f"请求 {request_id} 的 {kind} 决策出现两次"
                 f"——账本次序异常，fail-closed")
        seen.add(kind)
    if kind == "prefill" and isinstance(request_id, str):
        # （2026-10-01 清除）原"hit_states 判重"臂的前件严格弱于上方 F5
        # 判重（prefill 已入 seen_kinds，第二次 prefill 在那里先爆），
        # 永不可达——重复 prefill 的防线唯一化到 F5。
        hit_state, evidence = variant["hit_state"](
            decision, turns.get(request_id), request_id)
        state.hit_states[request_id] = (hit_state, evidence)
    extracted = variant["extract_events"](record, state)
    # extract_events 返回逐事件 dict（各仓映射代码零改动），此处统一转
    # 紧凑元组缓存（下标含义见 AdapterState；end_ns 恒 NA 不入缓存）。
    state.events.extend(
        (event["request_id"], event["start_ns"], event["bytes"],
         event["source"], event["target"], event["cause"])
        for event in extracted)
    if variant["shard_sum_check"]:
        # native 不变量复检：Σshards == total_bytes（以 native 为准）。
        for holder in (decision.get("prefill_decode_transfer"),
                       decision.get("history_transfer")):
            if isinstance(holder, dict):
                total = holder.get("total_bytes")
                shards = holder.get("shards")
                if (isinstance(total, int) and isinstance(shards, list)
                        and shards):
                    shard_total = _strict_shard_sum(shards, "shard_sum_check")
                    if shard_total != total:
                        state.shard_sum_violations += 1
        for field in ("completion_evictions", "history_evictions",
                      "prefill_evictions", "decode_evictions"):
            for holder in decision.get(field) or []:
                if isinstance(holder, dict):
                    total = holder.get("total_bytes")
                    shards = holder.get("shards")
                    if isinstance(total, int) and isinstance(shards, list):
                        shard_total = _strict_shard_sum(shards, "shard_sum_check")
                        if shard_total != total:
                            state.shard_sum_violations += 1


def cmd_adapter(args: argparse.Namespace) -> int:
    # CLI 入口（独立运行行为不变）。A4 driver 经 adapter_prepare /
    # adapter_consume / adapter_emit 组合复用同一逻辑（单遍 decision log）。
    # M6-11：REPO_VARIANTS 查找 + None 守卫只在 adapter_prepare 一处做
    # （原 CLI 路径的重复查找已删）。
    repo_variant = detect_repo_variant(args.run_dir, args.repo_variant)
    manifest = load_request_manifest(args.run_dir, args.request_manifest)
    variant, turns, state = adapter_prepare(repo_variant, manifest)
    log_path = args.run_dir / DECISION_LOG_RELPATH
    for record in iter_jsonl(log_path):
        adapter_consume(record, variant, turns, state)
    return adapter_emit(args, repo_variant, variant, manifest, turns, state)


def adapter_emit(args: argparse.Namespace, repo_variant: str, variant: dict,
                 manifest: dict, turns: dict[str, int],
                 adapter_state: AdapterState) -> int:
    if adapter_state.n_records == 0:
        # 空 decision log（文件在场但零记录）不再静默 exit 0 产出全空
        # 产物——与 hbm_watermark 的"没有任何可重放的 KV 动作"同款
        # fail-closed 口径。
        fail("decision log 零记录——空账本拒绝产出（fail-closed）")
    events = adapter_state.events
    hit_states = adapter_state.hit_states
    shard_sum_violations = adapter_state.shard_sum_violations
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
            ((f"a{index:06d}", event[0], event[1], NA, event[2], event[3],
              event[4], event[5])
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
                         out_path=out_path,
                         summary=summary)
    return 0


def reconcile(run_dir: Path, repo_variant: str,
              out_path: Optional[Path],
              summary: dict) -> int:
    """canonical bytes/count vs native 逐项对账（以 native 为准）。

    native 侧：独立从 decision log 重放聚合（按 family）；canonical 侧：
    重新解析已写出的 cache_events.csv 聚合。（原形参 events 在函数体内
    零使用——对账始终重读 CSV + 独立重放 native，2026-08-30 阶段2加固
    §3.4a 删除该死参数；原形参 variant 同样零使用，已删。）
    """
    import csv as _csv

    native: dict[str, dict[str, int]] = {}
    # F1（2026-09-30）：native 重放侧同步 legacy 去重——canonical 侧
    # （extract_events_face_wscllm 经 AdapterState）与 native 侧必须同
    # 键同跳，否则 B2 窗口产物两边一跳一不跳会造成假性对账失败。
    legacy_seen: dict[str, set] = {}
    for record in iter_jsonl(run_dir / DECISION_LOG_RELPATH):
        decision = record.get("decision") or {}
        request_id = record.get("request_id")
        kind = record.get("kind")
        tick = record.get("tick")
        buckets = []
        if kind == "prefill":
            transfer_list = decision.get("history_transfers")
            if isinstance(transfer_list, list):
                # B3-6（S2 新产物）/ B4（-LRU 新产物）：逐段传输对象
                # ——native 侧按段计数/求和（与 canonical 逐段事件同粒度对账）。
                for entry in transfer_list:
                    if not isinstance(entry, dict):
                        # canonical 侧同款 fail-closed（R5-batch3）：非
                        # dict 成员不得静默 continue——双侧同丢对账恒过。
                        fail(f"reconcile: history_transfers 成员必须是对象"
                             f"（{_row_where(record)}）——损坏行 fail-closed")
                    if entry.get("kind") == "local_hit":
                        continue
                    # bytes 提取式与 canonical 侧 _transfer_object_events
                    # 对称（total_bytes 非整数 → Σshards 回退）——单侧有
                    # 回退会把"有 shards 无 total_bytes"的边缘形状记成
                    # 假性对账失败。
                    total = entry.get("total_bytes")
                    if not isinstance(total, int):
                        shards = entry.get("shards")
                        if isinstance(shards, list):
                            total = _strict_shard_sum(shards, "reconcile prefill transfer shards")
                    if isinstance(total, int) and not isinstance(total, bool) and total > 0:
                        buckets.append((FAMILY_HISTORY, total))
            else:
                # legacy 标量回退（2026-10-01 对称修正）：只读
                # history_transfer_bytes 标量——canonical 侧
                # extract_events_face_wscllm 的同名回退只消费该标量，
                # 本侧此前额外优先消费单 dict history_transfer（sh_1.0
                # 形状），dict 与标量取值不同时两侧入账来源不同 → 假性
                # 对账失败（exit 2）。双侧同规：只认标量。
                nbytes = decision.get("history_transfer_bytes")
                if isinstance(nbytes, int) and not isinstance(nbytes, bool) and nbytes > 0:
                    buckets.append((FAMILY_HISTORY, nbytes))
            # 逐出字段单键取用（与 extract_events_face_wscllm 同判别式）
            # ：契约分段字段在场（B3+，含 S1 本名同形字段）消费分段行；
            # 缺席（B2 窗口/旧产物）消费 legacy union 字段——判别错级会把
            # B2 逐出整类丢失或 B3 双序列化双计。
            if "history_evictions" in decision \
                    or "prefill_evictions" in decision:
                eviction_fields = ("history_evictions", "prefill_evictions")
                seen_keys = None
            else:
                eviction_fields = ("admission_evictions",
                                   "decode_target_evictions")
                # R2-1 对称收口（2026-09-30）：键登记只在 legacy 回退分支做
                # （与 canonical 侧同口径）——契约路径不记键，避免混排行
                # 造成假性对账失败。
                seen_keys = legacy_seen.setdefault(str(request_id), set())
            for field in eviction_fields:
                for entry in decision.get(field) or []:
                    if not isinstance(entry, dict):
                        # canonical 侧同款 fail-closed（R5-batch3）：非
                        # dict 成员此前在 _eviction_bytes_face 的
                        # entry.get 上裸 AttributeError 逃出 fail-closed
                        # 出口码 2。
                        fail(f"reconcile: {field} 成员必须是对象"
                             f"（{_row_where(record)}）——损坏行 fail-closed")
                    # 写侧逐出行恒为 remote_store KVTransfer（local_hit 是
                    # 恢复动作且 history_transfers 恒落空元组）——旧实现
                    # 的 local_hit 过滤不可达，已删。
                    if seen_keys is not None:
                        key = _legacy_eviction_key(entry)
                        if key is not None:
                            seen_keys.add(key)
                    nbytes = _eviction_bytes_face(entry)
                    if not nbytes:
                        total = entry.get("total_bytes")
                        shards = entry.get("shards")
                        if not isinstance(total, int) and isinstance(
                                shards, list) and shards:
                            total = _strict_shard_sum(shards, "reconcile eviction shards")
                        nbytes = total if isinstance(total, int) else None
                    if nbytes:
                        buckets.append((FAMILY_EVICTION, nbytes))
        elif kind == "decode":
            # P→D 传输（2026-10-01 对称修正）：与 canonical 侧
            # extract_events_face_wscllm 的 decode 臂完全同规——dict 在场
            # 时只认 total_bytes（local_hit → 0，非整数 → 无事件），
            # 不再回落到 prefill_decode_transfer_bytes 标量（单侧回落会
            # 把"dict 在场但 total_bytes 非整数"的形状记成假性对账失败）；
            # dict 缺席才走标量回退。
            transfer = decision.get("prefill_decode_transfer")
            nbytes = None
            if isinstance(transfer, dict):
                if transfer.get("kind") == "local_hit":
                    nbytes = 0  # 名义 KV 规模，无物理搬运
                else:
                    nbytes = transfer.get("total_bytes")
            else:
                nbytes = decision.get("prefill_decode_transfer_bytes")
            if isinstance(nbytes, int) and not isinstance(nbytes, bool) and nbytes > 0:
                buckets.append((FAMILY_PD, nbytes))
            # 逐出单键取用（与 extract 同式）：decode_evictions 缺席回退
            # legacy decode_target_evictions（B2 起随 decode 行落盘）；
            # F1 去重：跳过 prefill 行准入快照已计的 #2 条目。
            if "decode_evictions" in decision:
                eviction_rows = decision.get("decode_evictions")
                seen_keys = None
            else:
                eviction_rows = decision.get("decode_target_evictions")
                seen_keys = legacy_seen.get(str(request_id))
            for entry in eviction_rows or []:
                if not isinstance(entry, dict):
                    # canonical 侧同款 fail-closed（R5-batch3）：非 dict
                    # 成员此前在 entry.get 上裸 AttributeError。
                    fail(f"reconcile: 逐出行成员必须是对象"
                         f"（{_row_where(record)}）——损坏行 fail-closed")
                if seen_keys:
                    key = _legacy_eviction_key(entry)
                    if key is not None and key in seen_keys:
                        continue
                total = entry.get("total_bytes")
                shards = entry.get("shards")
                if not isinstance(total, int) and isinstance(
                        shards, list) and shards:
                    total = _strict_shard_sum(shards, "reconcile decode shards")
                if isinstance(total, int) and not isinstance(total, bool) and total > 0:
                    buckets.append((FAMILY_EVICTION, total))
        elif kind == "completion":
            for entry in decision.get("completion_evictions") or []:
                if not isinstance(entry, dict):
                    fail(f"reconcile: completion_evictions 成员必须是对象"
                         f"（{_row_where(record)}）——损坏行此前被静默跳过"
                         f"（双侧同丢对账恒过），fail-closed")
                nbytes = _eviction_bytes_face(entry)
                total = entry.get("total_bytes")
                shards = entry.get("shards")
                if not nbytes:
                    if isinstance(total, int):
                        nbytes = total
                    elif isinstance(shards, list):
                        nbytes = _strict_shard_sum(shards, "reconcile completion shards")
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
