#!/usr/bin/env python3
"""Hop-Bytes = Σ_actions bytes × noc_hops（主规格 §1.5-A；纯后处理）。

从运行产物中"含 noc hops 的传输记录"聚合，分请求/总量两档输出。各仓
真实字段名已在基线产物中核对并登记于 REPO_HOP_SOURCES（先定位字段、
逐仓登记，不猜测）：

  astra-sim-sh_1.0  shard 级：decision log 的 history_transfer /
                    prefill_decode_transfer / 各类逐出传输对象的
                    shards[].noc_path（hops = len(noc_path)-1；生成侧
                    generate_face_trace.py:587 亦落 noc_hops 字段——
                    读取时优先用显式 noc_hops，缺省由 noc_path 推导）。
  astra-sim-wscllm  实例级：decode 决策的 static_route.hop_count ×
                    prefill_decode_transfer.total_bytes（粒度为实例间
                    hop，非 rank 级）。2026-08-26 B2wp9py 起 prefill 决策
                    对 history NOC_MIGRATE 迁移附加 noc_hops 字段
                    （实例图最短路，同粒度）→ history 迁移纳入覆盖
                    （60s 基线覆盖 70.4% → 满覆盖；字段为 A/B 对拍
                    剥离清单条目）。
                    2026-09-05 问题 4b 起切 shard 级 mesh 跳口径：decode
                    行 shards[] 与 prefill 行 history_transfer_shards[]
                    逐 shard bytes×noc_hops；旧产物（无路由字段）回退
                    上述聚合口径，数值不变（详见 REPO_HOP_
                    SOURCES notes）。
  astra-sim-face    per-TP-shard 级（B2wp9py 起决策日志附带只读
                    hops 列表：prefill=history_noc_hops、decode=
                    kv_noc_hops，与 shards 一一对齐；hops 全等时
                    聚合 bytes×hops[0] 与逐 shard 求和严格相等）。
                    旧产物无字段 → bytes_without_hops。
  astra-sim-sh_2.0  WP9-线5（2026-08-26）起 decision 逐传输对象序列化
                    transfer_hop_bytes[].{total_bytes, noc_hop_bytes,
                    shard_count}（KVTransfer shards 按 deterministic_xy_
                    route 生成、在线复用）→ hop_bytes 可聚合;local_hit
                    无物理搬运不计入。旧产物（字段缺失）回退 coverage=0。
  astra-sim-sh_3.0  B4 起（2026-08-26）消费 shards[].noc_hops/noc_path
                    （现状 completion_evictions 落盘；count/fallback 语义
                    同 S1）；无路由的聚合 bytes 字段不计入本账。

覆盖率为 0 的仓输出 hop_bytes=0 并显式标注 coverage，不臆造 hop 数。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slo_common import (  # noqa: E402
    DECISION_LOG_RELPATH, NA, detect_repo_variant, emit_json, fail,
    fmt_ratio, iter_jsonl, open_output, run_main, write_csv,
)

TOTAL_COLUMNS = ("repo_variant", "granularity", "actions_with_hops",
                 "hop_bytes_total", "bytes_with_hops", "bytes_without_hops",
                 "coverage_bytes_ratio", "notes")
PER_REQUEST_COLUMNS = ("request_id", "actions_with_hops", "hop_bytes",
                       "bytes_with_hops", "bytes_without_hops")


def _byte_count(value: Any) -> Optional[int]:
    """bytes 字段的严格解析（D2 修复 2026-09-28）。

    返回正整数或 None（畸形）。此前两处类型漏洞：
      * bool 冒充（isinstance(True, int) 为真）→ 整 shard 按 1 字节计；
      * 浮点 bytes（JSON number 150.0）→ 非 int 被跳过，整 shard 从
        分子分母同时蒸发。
    现口径：int 且 >0 原样；bool 一律 None；float 仅在整数值时按 int
    接受（150.0 == 150，零臆造）；其余类型/非正值 → None（调用方跳过，
    与本文件"宁缺勿造"的畸形条目处置一致）。
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float) and math.isfinite(value) \
            and value.is_integer() and value > 0:
        return int(value)
    return None


def _shard_hops(shard: dict) -> Optional[int]:
    """优先显式 noc_hops 字段；缺省由 noc_path 推导（len-1）；皆缺 → None。

    F2_PY 修复（2026-09-29，RV5-F8）：hops 判定族补 bool 排除——
    ``isinstance(True, int)`` 为真，bool hops 曾按 1 跳计入（True>=0 且
    参与 hop_bytes 乘法），与同文件 _byte_count 对 bytes 的 bool 一律
    拒绝口径不一致；现 bool（True/False）一律 None → 该条字节进
    bytes_without_hops（宁缺勿造，不臆造 hop 数）。

    本函数即 hopbytes 全模块唯一的路由 hops 解析实现：原 _route_hops 与
    本函数是 AST 级逐字重复（B2 HP-3 合并 collect_sh30 同款遗漏），现为
    单一实现 + 双入口名别名（REPO_HOP_SOURCES/调用方不受影响），见文件
    尾部 _route_hops = _shard_hops。
    """
    hops = shard.get("noc_hops")
    if isinstance(hops, int) and not isinstance(hops, bool) and hops >= 0:
        return hops
    path = shard.get("noc_path")
    if isinstance(path, list) and path:
        return max(0, len(path) - 1)
    return None


def _accumulate_shard_transfers(holder: dict, request_id: str, acc: dict,
                                per_request: dict) -> None:
    if not isinstance(holder, dict):
        return
    shards = holder.get("shards")
    if not isinstance(shards, list):
        return
    for shard in shards:
        if not isinstance(shard, dict):
            continue
        nbytes = _byte_count(shard.get("bytes"))
        if nbytes is None:
            continue
        hops = _shard_hops(shard)
        slot = per_request.setdefault(
            request_id, {"actions": 0, "hop_bytes": 0,
                         "bytes_with": 0, "bytes_without": 0})
        if hops is None:
            slot["bytes_without"] += nbytes
            acc["bytes_without_hops"] += nbytes
        else:
            slot["actions"] += 1
            slot["hop_bytes"] += nbytes * hops
            slot["bytes_with"] += nbytes
            acc["actions_with_hops"] += 1
            acc["hop_bytes_total"] += nbytes * hops
            acc["bytes_with_hops"] += nbytes


def collect_sh10(record: dict, acc: dict, per_request: dict) -> None:
    decision = record.get("decision") or {}
    request_id = record.get("request_id") or NA
    kind = record.get("kind")
    if kind == "prefill":
        _accumulate_shard_transfers(decision.get("history_transfer"),
                                    request_id, acc, per_request)
        for field in ("history_evictions", "prefill_evictions"):
            for entry in decision.get(field) or []:
                _accumulate_shard_transfers(entry, request_id, acc,
                                            per_request)
    elif kind == "decode":
        _accumulate_shard_transfers(decision.get("prefill_decode_transfer"),
                                    request_id, acc, per_request)
        for entry in decision.get("decode_evictions") or []:
            _accumulate_shard_transfers(entry, request_id, acc, per_request)
    elif kind == "completion":
        for entry in decision.get("completion_evictions") or []:
            _accumulate_shard_transfers(entry, request_id, acc, per_request)


def collect_face(record: dict, acc: dict, per_request: dict) -> None:
    """face 变体（2026-08-27 B3 补齐）：B2wp9py 起决策日志附带只读
    noc 观测字段——prefill 决策 history_noc_hops、decode 决策
    kv_noc_hops（均为 per-TP-shard hops 列表，与 shards 一一对齐，
    无物理传输时空列表）。基线事实核对（60s 窗）：decode 1244 条
    shards/hops 长度零错位、每条 hops 全等；prefill 1079 条 hops 全等。
    聚合口径：
      decode  = Σ shards[i].bytes × hops[i]（逐 shard 精确）；
      prefill = history_transfer_bytes × hops[0]（per-shard hops 全等
      ⇒ 与逐 shard bytes×hops 求和严格相等，不依赖 shard bytes 分布）。
    旧产物无字段 → bytes_without_hops（向后兼容，coverage 如实为低）。
    decode shards/hops 长度错位 → 整条回退 bytes_without_hops 并计
    hops_length_mismatch（2026-09-27 修：原实现对位条件不满足时整条
    静默丢弃，字节从 coverage 分母消失；60s 基线零错位，此为防御）。
    P2-20（2026-09-28）修：decode 行 kv_noc_hops 缺席/非列表（旧产物）
    同样整条回退 bytes_without_hops——原 isinstance 门两个分支都不过时
    整条传输字节从 coverage 分母蒸发（违反本 docstring 与 prefill 分支
    的回退承诺，coverage_bytes_ratio 虚高）。
    """
    decision = record.get("decision") or {}
    request_id = record.get("request_id") or NA
    kind = record.get("kind")

    def _slot():
        return per_request.setdefault(
            request_id, {"actions": 0, "hop_bytes": 0,
                         "bytes_with": 0, "bytes_without": 0})

    if kind == "prefill":
        nbytes = _byte_count(decision.get("history_transfer_bytes"))
        hops = decision.get("history_noc_hops")
        if nbytes is not None:
            # F2_PY 修复（2026-09-29，RV5-F8）：元素门加 bool 排除——
            # bool 元素冒充 1 跳（True==1）曾把整条字节按 hops[0]=1 计
            # 覆盖侧；现整条回退 bytes_without（宁缺勿造，同下分支）。
            if isinstance(hops, list) and hops and all(
                    isinstance(h, int) and not isinstance(h, bool)
                    and h >= 0 for h in hops):
                hop_bytes = nbytes * hops[0]
                slot = _slot()
                slot["actions"] += 1
                slot["hop_bytes"] += hop_bytes
                slot["bytes_with"] += nbytes
                acc["actions_with_hops"] += 1
                acc["hop_bytes_total"] += hop_bytes
                acc["bytes_with_hops"] += nbytes
            else:
                slot = _slot()
                slot["bytes_without"] += nbytes
                acc["bytes_without_hops"] += nbytes
    elif kind == "decode":
        transfer = decision.get("prefill_decode_transfer")
        hops = decision.get("kv_noc_hops")
        shards = (transfer or {}).get("shards") if isinstance(
            transfer, dict) else None
        if isinstance(shards, list) and isinstance(hops, list) \
                and len(shards) != len(hops):
            # shards/hops 长度错位：shard 级对位失效——回退按 bytes_without
            # 记账（宁缺勿造），防整条记录的字节静默消失使 coverage 分母
            # 失真；异常条数计入 summary（hops_length_mismatch）。
            # (2026-10-01) The mismatch EVENT count is decoupled from the
            # byte count: a misplaced row whose shard bytes are all
            # unparseable still counts as one mismatch event (the old
            # `if misplaced` gate silently under-reported such rows).
            for shard in shards:
                nbytes = _byte_count((shard or {}).get("bytes"))
                if nbytes is None:
                    continue
                slot = _slot()
                slot["bytes_without"] += nbytes
                acc["bytes_without_hops"] += nbytes
            # (2026-10-01, D1 复审) The former `misplaced` byte accumulator
            # became write-only when the event count above was decoupled
            # from it; removed.
            acc["hops_length_mismatch"] += 1
            return
        if isinstance(shards, list) and isinstance(hops, list):
            for shard, hop in zip(shards, hops):
                nbytes = _byte_count((shard or {}).get("bytes"))
                if nbytes is None:
                    continue
                # F2_PY 修复（2026-09-29，RV5-F8）：对位 hop 门加 bool
                # 排除（bool 冒充 1 跳同族）——bool hop 的 shard 字节进
                # 无覆盖侧。
                if isinstance(hop, int) and not isinstance(hop, bool) \
                        and hop >= 0:
                    slot = _slot()
                    slot["actions"] += 1
                    slot["hop_bytes"] += nbytes * hop
                    slot["bytes_with"] += nbytes
                    acc["actions_with_hops"] += 1
                    acc["hop_bytes_total"] += nbytes * hop
                    acc["bytes_with_hops"] += nbytes
                else:
                    slot = _slot()
                    slot["bytes_without"] += nbytes
                    acc["bytes_without_hops"] += nbytes
        elif isinstance(shards, list):
            # P2-20（2026-09-28）：kv_noc_hops 缺席/非列表（旧产物）——
            # 按字节原额计入 bytes_without_hops 分母（宁缺勿造），不再
            # 整条蒸发（与上行长度错位臂、prefill 分支同口径）。
            for shard in shards:
                nbytes = _byte_count((shard or {}).get("bytes"))
                if nbytes is None:
                    continue
                slot = _slot()
                slot["bytes_without"] += nbytes
                acc["bytes_without_hops"] += nbytes


# F2_PY 修复（2026-09-29，RV5-F8）：_route_hops 与 _shard_hops 是 AST 级
# 逐字重复的孪生助手（B2 HP-3 刚合并过 collect_sh30 的同款重复，这两个
# 却留下）——合并为单一实现（_shard_hops，含 bool 排除），双入口名保留
# （collect_wscllm 的 any() 探测与既有调用面不变），与 collect_sh30 =
# collect_sh10 先例同一手法。
_route_hops = _shard_hops


def collect_wscllm(record: dict, acc: dict, per_request: dict) -> None:
    decision = record.get("decision") or {}
    request_id = record.get("request_id") or NA
    if record.get("kind") != "decode":
        # 历史迁移（prefill 决策）：问题 4b（2026-09-05）起 shard 级
        # mesh 跳口径优先——decision.history_transfer_shards[] 逐 shard
        # bytes×noc_hops（生成侧序列化 noc_path/noc_hops，count/fallback
        # 语义同 S1 分支）；字段缺席（旧产物）回退 2026-08-26 B2wp9py
        # 的聚合口径 history_transfer_bytes×noc_hops（实例图最短路），
        # 再无该字段时按 bytes_without_hops 处理（向后兼容）。
        if record.get("kind") == "prefill":
            shard_rows = decision.get("history_transfer_shards")
            if isinstance(shard_rows, list) and shard_rows:
                _accumulate_shard_transfers({"shards": shard_rows},
                                            request_id, acc, per_request)
                return
            nbytes = _byte_count(decision.get("history_transfer_bytes"))
            if nbytes is not None:
                hops = decision.get("noc_hops")
                slot = per_request.setdefault(
                    request_id, {"actions": 0, "hop_bytes": 0,
                                 "bytes_with": 0, "bytes_without": 0})
                # F2_PY 修复（2026-09-29，RV5-F8）：聚合口径 hops 门加
                # bool 排除（同族冒充）。
                if isinstance(hops, int) and not isinstance(hops, bool) \
                        and hops >= 0:
                    slot["actions"] += 1
                    slot["hop_bytes"] += nbytes * hops
                    slot["bytes_with"] += nbytes
                    acc["actions_with_hops"] += 1
                    acc["hop_bytes_total"] += nbytes * hops
                    acc["bytes_with_hops"] += nbytes
                else:
                    slot["bytes_without"] += nbytes
                    acc["bytes_without_hops"] += nbytes
        return
    # decode：问题 4b（2026-09-05）shard 级优先——prefill_decode_transfer
    # 含非空 shards[] 且任一 shard 携带 noc_hops/noc_path 时逐 shard
    # bytes×hops（生成侧 _kv_transfer_dict(hardware=...) 落的路由字段；
    # 无路由 shard 计 bytes_without_hops，不臆造）；否则（旧产物/旧行为，
    # shards 无路由信息或字段缺席）回退聚合口径
    # total_bytes×static_route.hop_count（实例间 hop 粒度）。
    transfer = decision.get("prefill_decode_transfer")
    shards = (transfer or {}).get("shards") if isinstance(
        transfer, dict) else None
    if isinstance(shards, list) and shards and any(
            _route_hops(shard) is not None
            for shard in shards if isinstance(shard, dict)):
        _accumulate_shard_transfers(transfer, request_id, acc, per_request)
        return
    static_route = decision.get("static_route")
    nbytes = _byte_count(
        (transfer or {}).get("total_bytes") if isinstance(
            transfer, dict) else None)
    hops = (static_route or {}).get("hop_count") if isinstance(
        static_route, dict) else None
    if nbytes is None:
        return
    slot = per_request.setdefault(
        request_id, {"actions": 0, "hop_bytes": 0,
                     "bytes_with": 0, "bytes_without": 0})
    # F2_PY 修复（2026-09-29，RV5-F8）：static_route.hop_count 门加 bool
    # 排除（同族冒充——True 曾按 1 跳计覆盖侧）。
    if isinstance(hops, int) and not isinstance(hops, bool) and hops >= 0:
        slot["actions"] += 1
        slot["hop_bytes"] += nbytes * hops
        slot["bytes_with"] += nbytes
        acc["actions_with_hops"] += 1
        acc["hop_bytes_total"] += nbytes * hops
        acc["bytes_with_hops"] += nbytes
    else:
        slot["bytes_without"] += nbytes
        acc["bytes_without_hops"] += nbytes


def collect_aggregate_only(record: dict, acc: dict, per_request: dict) -> None:
    """S2 旧产物（transfer_hop_bytes 缺席，collect_sh20 回退）：
    bytes 全部计入无覆盖侧。"""
    decision = record.get("decision") or {}
    request_id = record.get("request_id") or NA
    kind = record.get("kind")
    nbytes = None
    if kind == "prefill":
        raw = decision.get("history_transfer_bytes")
        if isinstance(raw, int):
            nbytes = _byte_count(raw)
        else:
            transfer = decision.get("history_transfer")
            if isinstance(transfer, dict):
                nbytes = _byte_count(transfer.get("total_bytes"))
    elif kind == "decode":
        raw = decision.get("prefill_decode_transfer_bytes")
        if isinstance(raw, int):
            nbytes = _byte_count(raw)
        else:
            transfer = decision.get("prefill_decode_transfer")
            if isinstance(transfer, dict):
                nbytes = _byte_count(transfer.get("total_bytes"))
    if nbytes is not None:
        slot = per_request.setdefault(
            request_id, {"actions": 0, "hop_bytes": 0,
                         "bytes_with": 0, "bytes_without": 0})
        slot["bytes_without"] += nbytes
        acc["bytes_without_hops"] += nbytes


def collect_sh20(record: dict, acc: dict, per_request: dict) -> None:
    """S2：WP9-线5（2026-08-26）起 decision 附加 transfer_hop_bytes
    （逐传输对象 {total_bytes, noc_hop_bytes, shard_count}；shards 按
    deterministic_xy_route 生成，在线路径复用）。shard_count>0 的传输
    计入覆盖侧（hop_bytes 用决策时点序列化的 noc_hop_bytes，不再回读
    shards）；local_hit（shard_count=0，无物理搬运）不计入任何桶。
    D2 修复（2026-09-28）：noc_hop_bytes 缺席/非非负整数时该条字节不再
    计入覆盖侧（bytes_with_hops 虚增 coverage 而 hop_bytes 贡献 0 的
    "宣称覆盖"口径违反本文件 face 分支的 bytes_without 兑现承诺）——改
    按 bytes_without_hops 计（宁缺勿造）；noc_hop_bytes 为合法 int ≥0
    （含 0 跳，同实例交割）仍计覆盖侧。旧产物（字段缺失）回退
    collect_aggregate_only 语义（bytes 全部计入无覆盖侧），新旧产物可
    区分。"""
    decision = record.get("decision") or {}
    request_id = record.get("request_id") or NA
    kind = record.get("kind")
    relevant_kinds = ("prefill", "decode", "completion")
    hop_rows = decision.get("transfer_hop_bytes")
    if kind in relevant_kinds and isinstance(hop_rows, list):
        for row in hop_rows:
            if not isinstance(row, dict):
                continue
            nbytes = _byte_count(row.get("total_bytes"))
            hop_bytes = row.get("noc_hop_bytes")
            shard_count = row.get("shard_count")
            if (nbytes is None
                    or isinstance(shard_count, bool)
                    or not isinstance(shard_count, int)
                    or shard_count <= 0):
                continue  # local_hit / 名义规模：无物理搬运
            if isinstance(hop_bytes, int) and not isinstance(hop_bytes, bool) \
                    and hop_bytes >= 0:
                slot = per_request.setdefault(
                    request_id, {"actions": 0, "hop_bytes": 0,
                                 "bytes_with": 0, "bytes_without": 0})
                slot["actions"] += 1
                slot["bytes_with"] += nbytes
                slot["hop_bytes"] += hop_bytes
                acc["actions_with_hops"] += 1
                acc["bytes_with_hops"] += nbytes
                acc["hop_bytes_total"] += hop_bytes
            else:
                # 覆盖侧资格不成立（无可用 hop 数）：字节进无覆盖侧。
                slot = per_request.setdefault(
                    request_id, {"actions": 0, "hop_bytes": 0,
                                 "bytes_with": 0, "bytes_without": 0})
                slot["bytes_without"] += nbytes
                acc["bytes_without_hops"] += nbytes
        return
    collect_aggregate_only(record, acc, per_request)


# D2 修复（2026-09-28）：collect_sh30 与 collect_sh10 是逐句（AST 级）
# 相同的重复实现——合并为单一实现，两个入口名都保留（REPO_HOP_SOURCES
# 的 S1/S3 登记与任何外部调用不受影响）。S3 原口径说明（原 docstring）：
# S3（B4 升级，2026-08-26）消费 decision log 的 shards[].noc_hops/noc_path
# 新字段（B3 60s 证据：completion_evictions 96 行 / 960 shards 全带
# noc_hops；ad-hoc 复核 hop_bytes=2383397232640）。count/fallback 语义与
# S1 分支一致（_accumulate_shard_transfers：显式 noc_hops 优先、noc_path
# 推导兜底、无路由 bytes 计 bytes_without_hops）。holder 集照 S1
# （history_transfer / prefill_decode_transfer / 各类逐出列表）；S3 现仅
# completion_evictions 落 shards，其余 holder 留作字段同步后的前向兼容。
# B3 之前的 aggregate-only 停用口径（history_transfer_bytes /
# prefill_decode_transfer_bytes 无路由可聚）不计入本账——那些记录无 noc
# 路由信息，不属于 Hop-Bytes 可判定宇宙；其量级（60s 参考跑
# 1045595684864 B）由门 JSON 另行登记。
collect_sh30 = collect_sh10


REPO_HOP_SOURCES: dict[str, dict] = {
    "astra-sim-sh_1.0": {
        "collector": collect_sh10,
        "granularity": "shard(noc_path)",
        "notes": "shards[].noc_path/noc_hops 已落盘 decision log",
    },
    "astra-sim-wscllm": {
        "collector": collect_wscllm,
        "granularity": "shard(noc_path:decode/prefill)"
                       "+fallback-instance(static_route.hop_count)",
        "notes": "问题 4b（2026-09-05）起 shard 级 mesh 跳口径：decode 行 "
                 "prefill_decode_transfer.shards[] 与 prefill 行 "
                 "history_transfer_shards[] 逐 shard bytes×noc_hops"
                 "（生成侧 _kv_transfer_dict(hardware=)/调度器序列化 "
                 "noc_path/noc_hops；count/fallback 语义同 S1）。字段"
                 "缺席或无路由（旧产物）回退聚合口径（decode="
                 "total_bytes×static_route.hop_count、prefill="
                 "history_transfer_bytes×noc_hops，实例级粒度），旧"
                 "产物数值不变。",
    },
    "astra-sim-face": {
        "collector": collect_face,
        "granularity": "per-TP-shard(history_noc_hops/kv_noc_hops)",
        "notes": "B2wp9py 起决策日志附带只读 noc hops 列表"
                 "（prefill=history_noc_hops，decode=kv_noc_hops，"
                 "与 shards 一一对齐）；旧产物无字段 → "
                 "bytes_without_hops（coverage 如实降低，不臆造）",
    },
    "astra-sim-sh_2.0": {
        "collector": collect_sh20,
        "granularity": "shard-route(noc_hop_bytes)",
        "notes": "WP9-线5（2026-08-26）起 decision 逐传输对象序列化 "
                 "noc_hop_bytes（shards 按 deterministic_xy_route 生成，"
                 "决策时点即知）；local_hit 无物理搬运不计入；旧产物"
                 "（字段缺失）回退 coverage=0 口径",
    },
    "astra-sim-sh_3.0": {
        "collector": collect_sh30,
        "granularity": "shard(noc_path)",
        "notes": "B4 起消费 shards[].noc_hops/noc_path（现状仅 "
                 "completion_evictions 落盘；其余 holder 前向兼容）；"
                 "无路由的聚合 bytes 字段不计入本账（count/fallback "
                 "语义同 S1）",
    },
}


def hopbytes_prepare(repo_variant: str) -> tuple[dict, dict, dict]:
    """A4 driver 用：source 登记 + 空累积态。"""
    source = REPO_HOP_SOURCES.get(repo_variant)
    if source is None:
        fail(f"未登记的 repo_variant：{repo_variant}（REPO_HOP_SOURCES 需扩表）")
    acc = {"actions_with_hops": 0, "hop_bytes_total": 0,
           "bytes_with_hops": 0, "bytes_without_hops": 0,
           "hops_length_mismatch": 0}
    return source, acc, {}


def cmd_hopbytes(args: argparse.Namespace) -> int:
    # CLI 入口（独立运行行为不变）。A4 driver 经 hopbytes_prepare /
    # collector / hopbytes_emit 组合复用同一逻辑（单遍 decision log）。
    repo_variant = detect_repo_variant(args.run_dir, args.repo_variant)
    source, acc, per_request = hopbytes_prepare(repo_variant)
    for record in iter_jsonl(args.run_dir / DECISION_LOG_RELPATH):
        source["collector"](record, acc, per_request)
    return hopbytes_emit(args, repo_variant, source, acc, per_request)


def hopbytes_emit(args: argparse.Namespace, repo_variant: str, source: dict,
                  acc: dict, per_request: dict[str, dict]) -> int:
    total_bytes = acc["bytes_with_hops"] + acc["bytes_without_hops"]
    coverage = (acc["bytes_with_hops"] / total_bytes) if total_bytes else 0.0
    total_row = (repo_variant, source["granularity"],
                 acc["actions_with_hops"], acc["hop_bytes_total"],
                 acc["bytes_with_hops"], acc["bytes_without_hops"],
                 fmt_ratio(coverage), source["notes"])
    stream, close = open_output(args.output, "slo_hopbytes_total.csv",
                                args.run_dir)
    try:
        write_csv(stream, TOTAL_COLUMNS, [total_row])
    finally:
        if close:
            stream.close()
    pr_stream, pr_close = open_output(args.per_request,
                                      "slo_hopbytes_per_request.csv",
                                      args.run_dir)
    try:
        write_csv(
            pr_stream, PER_REQUEST_COLUMNS,
            ((rid, slot["actions"], slot["hop_bytes"], slot["bytes_with"],
              slot["bytes_without"])
             for rid, slot in sorted(per_request.items())))
    finally:
        if pr_close:
            pr_stream.close()
    summary = {
        "command": "hopbytes",
        "repo_variant": repo_variant,
        "formula": "Hop-Bytes = sum(bytes * noc_hops) over transfer "
                   "records carrying hop fields (spec 1.5-A)",
        **acc,
        "coverage_bytes_ratio": coverage,
        "notes": source["notes"],
        "granularity": source["granularity"],
    }
    emit_json(sys.stderr, summary)
    if coverage == 0.0:
        print(f"[hopbytes] 警告：{repo_variant} 产物无可用的 noc hops 字段"
              f"（coverage=0）——Hop-Bytes=0 仅表示无覆盖，见 notes",
              file=sys.stderr)
    if args.json:
        jstream, jclose = open_output(args.json, "slo_hopbytes.json",
                                      args.run_dir)
        try:
            emit_json(jstream, summary)
        finally:
            if jclose:
                jstream.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="hopbytes.py",
        description="Hop-Bytes = Σ bytes×noc_hops（分请求/总量两档；"
                    "仓内产物无 hops 字段时 coverage=0 并显式标注）")
    parser.add_argument("run_dir", type=Path,
                        help="运行目录（含 results/online_decision_log.jsonl）")
    parser.add_argument("-o", "--output", default="",
                        help="总量 CSV 输出（'-'=stdout；缺省写 "
                             "run_dir/slo_hopbytes_total.csv）")
    parser.add_argument("--per-request", default="",
                        help="分请求 CSV 输出（同上；缺省写 "
                             "run_dir/slo_hopbytes_per_request.csv）")
    parser.add_argument("--json", default="",
                        help="汇总 JSON 输出路径（可选）")
    parser.add_argument("--repo-variant", default=None,
                        help="显式指定 repo_variant（默认读 cpp.log init 行）")
    args = parser.parse_args()
    return int(cmd_hopbytes(args) or 0)


if __name__ == "__main__":
    sys.exit(run_main(main))
