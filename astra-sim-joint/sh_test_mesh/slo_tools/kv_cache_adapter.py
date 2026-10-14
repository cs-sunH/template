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
                   local_hit 无数据 shard（见 generate_face_trace.py 的 _emit_kv_transfer），
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
            # R7-F9（2026-09-29 face 移植）：重复 request_id 拒绝静默取后
            # 值（与 slo_stats cmd_session / hbm_watermark load_token_
            # manifest 的唯一性 fail-closed 同口径，消除"取后值/报错/覆盖"
            # 三口径并存）。
            if request_id in result:
                fail(f"per-request manifest request_id 重复：{request_id}")
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
    if not isinstance(transfer, dict):
        # 非 dict 真值（str/int/bool 等）旧行为裸 AttributeError → 退出码
        # 1，与 fail-closed 2 的失败语义契约不符（批次 D 镜像，2026-09-30）。
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


def _hit_state_joint(decision: dict, turn: Optional[int],
                     request_id: str) -> tuple[str, str]:
    """joint：按 R12 披露的 joint_action/history_tokens_before 映射三态。

    2026-10-01 审查轮修正：joint 行的 prefill_affinity_reason 恒为
    ``joint_<action>``（stay/copy/recompute——三动作，2026-10-14 远读
    退役），沿用 S3 五词表会把全部 joint 请求判成 not_supported、命中
    率输出主动错误的 0.0。映射语义（只读本行披露字段，不猜测）：

    - history_tokens_before == 0 → no_history（turn-0 首请求，无历史
      可命中，不计入 miss）；
    - joint_stay → full（历史驻留执行实例，本地全量命中）；
    - joint_copy → full（基础历史完整拷贝到执行实例，S1 noc_migrate
      同款口径——迁移非重算，命中语义看结果驻留）；
    - joint_recompute → miss（历史弃用重算，词表原生支持 miss 态；
      S1-S3 无 recompute 故 miss 从未出现，joint 下属预期）。

    joint_action/history_tokens_before 任一缺失（旧产物/审计行）保持
    not_supported 不猜测；字段在场但类型异常 fail-closed（退出码 2）。"""
    reason = decision.get("prefill_affinity_reason")
    action = decision.get("joint_action")
    before = decision.get("history_tokens_before")
    evidence = (f"prefill_affinity_reason={reason};joint_action={action};"
                f"history_tokens_before={before}")
    if action is None or before is None:
        return "not_supported", evidence
    if not isinstance(action, str):
        fail(f"joint_action 非字符串（request={request_id!r}，实得 "
             f"{action!r}）——形状谎报，fail-closed")
    if isinstance(before, bool) or not isinstance(before, int):
        fail(f"history_tokens_before 非整数（request={request_id!r}，实得 "
             f"{before!r}）——形状谎报，fail-closed")
    if before == 0:
        return "no_history", evidence
    mapping = {
        "joint_stay": "full",
        "joint_copy": "full",
        "joint_recompute": "miss",
    }
    if action not in mapping:
        # prefill_affinity_reason 形如 joint_<action>，但 joint_action 走
        # 出三动作词表（未来动作面扩展）——不猜测，留 not_supported。
        return "not_supported", evidence
    return mapping[action], evidence


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


def _shards_bytes_sum(shards, where: str) -> int:
    """shards[].bytes 的统一严格求和 choke point（M15/R2-F2，2026-09-29
    face 移植）。

    canonical 提取、shard_sum 复检与 --reconcile native 重放的每个
    shards 求和点都收敛到本函数：逐 shard 必须是对象且 bytes 为非负
    int（bool/float/str 一律 fail-closed）——int() 会把 float 静默截断
    记账、str 裸 TypeError 以退出码 1 崩壳。缺 bytes 键按 0 计（缺席
    语义，与"在场非法"区分）。
    """
    total = 0
    for shard in shards:
        if not isinstance(shard, dict):
            fail(f"{where}: shards 成员必须是对象（实得 {shard!r}）")
        value = shard.get("bytes")
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) \
                or value < 0:
            fail(f"{where}: shards[].bytes 含非法值 {value!r}"
                 f"（须为非负整数）——float 静默截断/str 裸崩均禁止，"
                 f"fail-closed")
        total += value
    return total


def _strict_total_bytes(value, where: str) -> int:
    """在场 total_bytes 的严格标量校验：bool/float/str 拒绝，负值拒绝
    （R7-F3，2026-09-29 face 移植：负 bytes 不再静默写入 canonical
    CSV——仓内其他 bytes 入账点均拒负）。"""
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{where}: total_bytes 在场但不是整数（实得 {value!r}）"
             f"——float 截断/str 裸崩禁止，fail-closed")
    if value < 0:
        fail(f"{where}: total_bytes 为负（{value}）——bytes 入账拒绝负值")
    return value


def _eviction_bytes_face(entry: dict) -> Optional[int]:
    """FACE/W 逐出条目 → 实例合计 bytes。

    本文件全部逐出求和 choke point（R8-F2，2026-09-29 face 移植）：
    canonical 提取与 --reconcile native 重放两侧共用（此处收紧即双侧
    同封，对账不再同盲）；canonical 侧的回退链与 reconcile native 臂
    同款，口径唯一。

    字节来源回退链：shard_bytes 列表严格求和 → 缺席时 total_bytes 严格
    标量（R7-F2 回退）→ 再缺席时 shards[].bytes 严格求和（走
    _shards_bytes_sum）→ 全缺席 None。校验强度与
    hbm_watermark._sum_shard_bytes 同型（逐元素非负整数，违反
    fail-closed exit 2）：float 走 int() 会静默截断记账、str/None 会裸
    TypeError 以退出码 1 崩壳。
    """
    if not isinstance(entry, dict):
        fail(f"逐出条目必须是对象（实得 {entry!r}）")
    where = (f"逐出条目（request={entry.get('trigger_request_id')!r}，"
             f"reason={entry.get('reason')!r}）")
    shard_bytes = entry.get("shard_bytes")
    if shard_bytes is not None and not isinstance(shard_bytes, list):
        fail(f"{where}: shard_bytes 在场但不是数组（实得 {shard_bytes!r}）"
             f"——fail-closed")
    if isinstance(shard_bytes, list) and shard_bytes:
        total = 0
        for value in shard_bytes:
            if isinstance(value, bool) or not isinstance(value, int) \
                    or value < 0:
                fail(f"{where}: shard_bytes 含非法元素 {value!r}"
                     f"（须为非负整数）——float 静默截断/str 裸崩"
                     f"均禁止（与 hbm_watermark._sum_shard_bytes 同强度，"
                     f"fail-closed）")
            total += value
        return total
    total = entry.get("total_bytes")
    if total is not None:
        return _strict_total_bytes(total, where)
    shards = entry.get("shards")
    if isinstance(shards, list) and shards:
        return _shards_bytes_sum(shards, where)
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
    """S1 传输对象 → canonical 事件（bytes 取 total_bytes，缺失则 Σshards）。

    bytes 口径收敛（M15/R2-F2 + R7-F3，2026-09-29 face 移植）：Σshards
    走 _shards_bytes_sum（严格求和 choke point）；total_bytes 在场须为
    非负 int（负值不再静默写入 canonical CSV——仓内其他 bytes 入账点
    均拒负；float/str 同拒），缺席且无 shards → 无事件（零字节传输不
    产行，语义不变）。
    """
    if not isinstance(transfer, dict):
        return []
    if transfer.get("kind") == "local_hit":
        return []
    where = (f"{family} 传输对象（request={default_request!r}，tick={tick}，"
             f"kind={transfer.get('kind')!r}）")
    total = transfer.get("total_bytes")
    if total is not None:
        nbytes = _strict_total_bytes(total, where)
    else:
        shards = transfer.get("shards")
        nbytes = (_shards_bytes_sum(shards, where)
                  if isinstance(shards, list) else 0)
    if nbytes <= 0:
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
    if not isinstance(tick, int):
        fail(f"decision log 行缺整数 tick（request={request_id!r}）")
    if kind == "prefill":
        transfers = decision.get("history_transfers")
        if isinstance(transfers, list):
            for entry in transfers:
                events.extend(_transfer_object_events(
                    entry, request_id, tick, FAMILY_HISTORY))
        else:
            nbytes = decision.get("history_transfer_bytes")
            if isinstance(nbytes, int) and nbytes > 0:
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
        if isinstance(nbytes, int) and nbytes > 0:
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


def extract_events_joint(record: dict) -> list[dict]:
    """joint 仓 canonical 事件提取（2026-10-01 审查轮；R2 补 prefill 对称）。

    S3 映射时代（2026-09-14 登记）沿用 extract_events_sh30，前提是
    "joint 的 prefill/decode/completion 行 schema 同构（S3 超集，P==D
    无迁移）"。该前提已被两处打破并在此补齐：

    * decode（D18 起 P 与 D 可跨实例）：prefill_decode_transfer（P→D
      迁移）与 decode_evictions；
    * prefill（R2）：joint 的 history_transfers 为**逐腿列表**（copy 逐
      块/后缀恢复逐组多腿为常态）、且携带四个逐出字段——S3 聚合单数
      口径使 canonical 与 native 臂 FAMILY_HISTORY 计数与
      FAMILY_EVICTION 全族失配（--reconcile 假红、无 reconcile 时
      cache_events.csv 静默漏记逐出族）。

    本函数逐分支镜像 reconcile native 臂的同一 choke point
    （_transfer_object_events 的 total_bytes→Σshards 回退链与严格校验；
    逐出走 _eviction_bytes_face 的 shard_bytes→total_bytes→Σshards
    链），两臂数值逐事件相等。"""
    kind = record.get("kind")
    decision = record.get("decision") or {}
    request_id = record.get("request_id")
    tick = record.get("tick")
    events: list[dict] = []
    if kind == "prefill":
        transfer_list = decision.get("history_transfers")
        if isinstance(transfer_list, list):
            for entry in transfer_list:
                events.extend(_transfer_object_events(
                    entry, request_id, tick, FAMILY_HISTORY))
        else:
            nbytes = decision.get("history_transfer_bytes")
            transfer = decision.get("history_transfer")
            if isinstance(transfer, dict):
                derived = _transfer_object_events(
                    transfer, request_id, tick, FAMILY_HISTORY)
                nbytes = derived[0]["bytes"] if derived else 0
            if isinstance(nbytes, int) and nbytes > 0:
                events.append(_event(
                    request_id, tick, nbytes,
                    decision.get("history_source_instance_index"),
                    decision.get("prefill_instance_index"),
                    f"{FAMILY_HISTORY}:reason="
                    f"{decision.get('prefill_affinity_reason')}"))
        for field in ("admission_evictions", "decode_target_evictions",
                      "history_evictions", "prefill_evictions"):
            for entry in decision.get(field) or []:
                if isinstance(entry, dict) and entry.get(
                        "kind") == "local_hit":
                    continue
                nbytes = _eviction_bytes_face(entry)
                if nbytes:
                    events.append(_event(
                        entry.get("trigger_request_id") or request_id,
                        tick, nbytes,
                        entry.get("source_instance_index"),
                        entry.get("target_instance_index"),
                        f"{FAMILY_EVICTION}:{entry.get('kind')}:"
                        f"{entry.get('reason')}"))
    elif kind == "decode":
        transfer = decision.get("prefill_decode_transfer")
        if isinstance(transfer, dict):
            events.extend(_transfer_object_events(
                transfer, request_id, tick, FAMILY_PD))
        else:
            nbytes = decision.get("prefill_decode_transfer_bytes")
            if isinstance(nbytes, int) and nbytes > 0:
                events.append(_event(
                    request_id, tick, nbytes,
                    decision.get("prefill_instance_index"),
                    decision.get("decode_instance_index"),
                    f"{FAMILY_PD}:reason="
                    f"{decision.get('prefill_affinity_reason')}"))
        for entry in decision.get("decode_evictions") or []:
            if isinstance(entry, dict) and entry.get("kind") != "local_hit":
                nbytes = _eviction_bytes_face(entry)
                if nbytes:
                    events.append(_event(
                        entry.get("trigger_request_id") or request_id,
                        tick, nbytes,
                        entry.get("source_instance_index"),
                        entry.get("target_instance_index"),
                        f"{FAMILY_EVICTION}:{entry.get('kind')}:"
                        f"{entry.get('reason')}"))
    elif kind == "completion":
        events.extend(extract_events_sh30(record))
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
                 "balance 80/resident_prefix_layers 19（→partial）；代码见 "
                 "sh30_online_scheduler.py 的 prefill 决策行构造",
    },
    "astra-sim-joint": {
        "hit_state": _hit_state_joint,
        "extract_events": extract_events_joint,
        "shard_sum_check": False,
        "partial_semantics": True,
        "notes": "2026-10-01 审查轮：hit_state 按 joint_action 三动作映射"
                 "（stay/copy→full、recompute→miss、"
                 "history_tokens_before==0→no_history——R12 披露字段直读；"
                 "2026-10-14 远读退役后 partial 态仅由池驻留词表承载）；"
                 "事件提取补 decode 分支（D18 起 P→D 可跨实例，decode 行携"
                 "带 prefill_decode_transfer/decode_evictions，S3 采集器无"
                 "此分支会漏记 PD/EVICTION 族且 --reconcile 假红）。"
                 "kind=joint_admission 审计行不消费",
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

    __slots__ = ("events", "hit_states", "shard_sum_violations",
                 "n_records")

    def __init__(self) -> None:
        self.events: list[tuple] = []
        self.hit_states: dict[str, tuple[str, str]] = {}
        self.shard_sum_violations = 0
        self.n_records = 0


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
    if record.get("kind") == "prefill" and isinstance(request_id, str):
        if request_id in state.hit_states:
            # R2-F4（2026-09-29 face 移植）：重复 prefill 决策拒绝静默
            # 覆盖 hit_state（同链消费者 load_imbalance 的 prefill/decode
            # 查重与 hbm_watermark seen_kinds 的"恰一次"不变量同口径，
            # fail-closed）。
            fail(f"请求 {request_id} 出现多次 prefill 决策（准入应恰一次）"
                 f"——kv_hit_state 不允许静默覆盖")
        hit_state, evidence = variant["hit_state"](
            decision, turns.get(request_id), request_id)
        state.hit_states[request_id] = (hit_state, evidence)
    extracted = variant["extract_events"](record)
    # extract_events 返回逐事件 dict（各仓映射代码零改动），此处统一转
    # 紧凑元组缓存（下标含义见 AdapterState；end_ns 恒 NA 不入缓存）。
    state.events.extend(
        (event["request_id"], event["start_ns"], event["bytes"],
         event["source"], event["target"], event["cause"])
        for event in extracted)
    if variant["shard_sum_check"]:
        # native 不变量复检：Σshards == total_bytes（以 native 为准）。
        # Σshards 走统一严格求和（M15/R2-F2，2026-09-29 face 移植：
        # int() 的 float 截断/str 裸崩不再绕过 choke point）。
        sum_where = (f"shard 复检（request={request_id!r}，"
                     f"kind={record.get('kind')}）")
        for holder in (decision.get("prefill_decode_transfer"),
                       decision.get("history_transfer")):
            if isinstance(holder, dict):
                total = holder.get("total_bytes")
                shards = holder.get("shards")
                if (isinstance(total, int) and isinstance(shards, list)
                        and shards):
                    if _shards_bytes_sum(shards, sum_where) != total:
                        state.shard_sum_violations += 1
        for field in ("completion_evictions", "history_evictions",
                      "prefill_evictions", "decode_evictions"):
            for holder in decision.get(field) or []:
                if isinstance(holder, dict):
                    total = holder.get("total_bytes")
                    shards = holder.get("shards")
                    if isinstance(total, int) and isinstance(shards, list):
                        if _shards_bytes_sum(shards, sum_where) != total:
                            state.shard_sum_violations += 1


def cmd_adapter(args: argparse.Namespace) -> int:
    # CLI 入口（独立运行行为不变）。A4 driver 经 adapter_prepare /
    # adapter_consume / adapter_emit 组合复用同一逻辑（单遍 decision log）。
    repo_variant = detect_repo_variant(args.run_dir, args.repo_variant)
    variant = REPO_VARIANTS.get(repo_variant)
    if variant is None:
        fail(f"未登记的 repo_variant：{repo_variant}（REPO_VARIANTS 需扩表，"
             f"禁止猜测映射）")
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
        # fail-closed 口径（批次 D 镜像，2026-09-30）。
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
                         variant=variant, out_path=out_path,
                         summary=summary)
    return 0


def reconcile(run_dir: Path, repo_variant: str, variant: dict,
              out_path: Optional[Path],
              summary: dict) -> int:
    """canonical bytes/count vs native 逐项对账（以 native 为准）。

    native 侧：独立从 decision log 重放聚合（按 family）；canonical 侧：
    重新解析已写出的 cache_events.csv 聚合。（原形参 events 在函数体内
    零使用——对账始终重读 CSV + 独立重放 native，2026-08-30 阶段2加固
    §3.4a 删除该死参数。）
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
            transfer_list = decision.get("history_transfers")
            if isinstance(transfer_list, list):
                # B3-6（S2 新产物）：partial 两段式恢复逐段对象——native
                # 侧按段计数/求和（与 canonical 逐段事件同粒度对账）。
                for entry in transfer_list:
                    # R7-F2（2026-09-29 face 移植）：native 臂与 canonical
                    # 臂同函数取 bytes——回退链/严格校验同封。
                    derived = _transfer_object_events(
                        entry, str(record.get("request_id") or ""),
                        record.get("tick", 0), FAMILY_HISTORY)
                    for event in derived:
                        buckets.append((FAMILY_HISTORY, event["bytes"]))
            else:
                nbytes = decision.get("history_transfer_bytes")
                transfer = decision.get("history_transfer")
                if isinstance(transfer, dict):
                    # R7-F2：同上——total_bytes 缺席时 Σshards 回退 + 负值
                    # /类型严格校验同封（此前 native 臂缺回退，
                    # --reconcile 假红）。
                    derived = _transfer_object_events(
                        transfer, str(record.get("request_id") or ""),
                        record.get("tick", 0), FAMILY_HISTORY)
                    nbytes = derived[0]["bytes"] if derived else 0
                if isinstance(nbytes, int) and nbytes > 0:
                    buckets.append((FAMILY_HISTORY, nbytes))
            for field in ("admission_evictions", "decode_target_evictions",
                          "history_evictions", "prefill_evictions"):
                for entry in decision.get(field) or []:
                    if isinstance(entry, dict) and entry.get(
                            "kind") == "local_hit":
                        continue
                    # 逐出 bytes 与 canonical 臂同 choke point（R8-F2，
                    # 2026-09-29 face 移植：shard_bytes→total_bytes→
                    # Σshards 回退链口径唯一，canonical 不再静默丢事件、
                    # --reconcile 不再假红）。
                    nbytes = _eviction_bytes_face(entry)
                    if nbytes:
                        buckets.append((FAMILY_EVICTION, nbytes))
        elif kind == "decode":
            transfer = decision.get("prefill_decode_transfer")
            nbytes = decision.get("prefill_decode_transfer_bytes")
            if isinstance(transfer, dict):
                # R7-F2：同 prefill 臂——native 与 canonical 同函数取
                # bytes（回退链/严格校验同封）。
                derived = _transfer_object_events(
                    transfer, str(record.get("request_id") or ""),
                    record.get("tick", 0), FAMILY_PD)
                nbytes = derived[0]["bytes"] if derived else 0
            if isinstance(nbytes, int) and nbytes > 0:
                buckets.append((FAMILY_PD, nbytes))
            for entry in decision.get("decode_evictions") or []:
                if isinstance(entry, dict) and entry.get(
                        "kind") != "local_hit":
                    nbytes = _eviction_bytes_face(entry)
                    if nbytes:
                        buckets.append((FAMILY_EVICTION, nbytes))
        elif kind == "completion":
            for entry in decision.get("completion_evictions") or []:
                if isinstance(entry, dict) and entry.get(
                        "kind") == "local_hit":
                    continue
                nbytes = _eviction_bytes_face(entry)
                if nbytes:
                    buckets.append((FAMILY_EVICTION, nbytes))
        for family, nbytes in buckets:
            slot = native.setdefault(family, {"count": 0, "bytes": 0})
            slot["count"] += 1
            slot["bytes"] += nbytes

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
