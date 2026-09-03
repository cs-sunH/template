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
                    2026-09-02 B4a 起 relevant_distributed 变体扩登记
                    1000/3100/3300 三族：kv_scatter / kv_remote_reads /
                    history_pull 类决策行携带生成侧 routes 元数据
                    （B2 发射函数返回结构，逐 route 行 bytes +
                    noc_hops/noc_path），逐行 bytes×noc_hops 聚合、
                    实例级口径；无 hops 数据不臆造（计
                    bytes_without_hops）。
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
import sys
from pathlib import Path
from typing import Optional

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


def _shard_hops(shard: dict) -> Optional[int]:
    """优先显式 noc_hops 字段；缺省由 noc_path 推导（len-1）。"""
    hops = shard.get("noc_hops")
    if isinstance(hops, int) and hops >= 0:
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
        nbytes = shard.get("bytes")
        if not isinstance(nbytes, int) or nbytes <= 0:
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
    """
    decision = record.get("decision") or {}
    request_id = record.get("request_id") or NA
    kind = record.get("kind")

    def _slot():
        return per_request.setdefault(
            request_id, {"actions": 0, "hop_bytes": 0,
                         "bytes_with": 0, "bytes_without": 0})

    if kind == "prefill":
        nbytes = decision.get("history_transfer_bytes")
        hops = decision.get("history_noc_hops")
        if isinstance(nbytes, int) and nbytes > 0:
            if isinstance(hops, list) and hops and all(
                    isinstance(h, int) and h >= 0 for h in hops):
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
                and len(shards) == len(hops):
            for shard, hop in zip(shards, hops):
                nbytes = (shard or {}).get("bytes")
                if not isinstance(nbytes, int) or nbytes <= 0:
                    continue
                if isinstance(hop, int) and hop >= 0:
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


RELEVANT_ROUTE_CATEGORIES = (1000, 3100, 3300)
# B4b/2026-09-02 按 B3 实际决策行对齐：1000 族路由在 prefill 行
# decision.history_pull_routes（直接 list，B3 序列化实际键）；其余防御性
# 键保留（dict.routes 形态）。3100/3300 在 kv_scatter/kv_remote_reads 行
# 的 decision 顶层 routes（已由顶层扫描命中）。
RELEVANT_ROUTE_HOLDERS = (
    "history_pull_routes", "kv_scatter", "kv_remote_reads",
    "kv_history_pull", "history_pull",
)


def _route_hops(row: dict) -> Optional[int]:
    """优先显式 noc_hops；缺省由 noc_path 推导（len-1）；皆缺 → None。"""
    hops = row.get("noc_hops")
    if isinstance(hops, int) and hops >= 0:
        return hops
    path = row.get("noc_path")
    if isinstance(path, list) and path:
        return max(0, len(path) - 1)
    return None


def _accumulate_relevant_routes(decision: dict, request_id: str,
                                acc: dict, per_request: dict) -> bool:
    """relevant_distributed 变体三族路由账（B4a/2026-09-02 扩登记）。

    从决策行的 kv_scatter / kv_remote_reads / history_pull 类 holder 读
    生成侧 routes 元数据——结构以 B2 发射函数（emit_piece_scatter /
    emit_remote_reads / emit_history_pulls）返回的 routes 为准：逐 route
    行 {category, bytes, noc_path, noc_hops, ...}，category ∈
    {1000 历史拉回, 3100 散布写, 3300 decode 远程读}；3300 行自带
    request_id（per 列车×成员×源），优先用行内值归账。

    聚合口径：实例级（hop 数为发射时点 XY 路由的真实跳数，与既有
    wscllm 实例级口径同族）；无 hops 数据不臆造——noc_hops/noc_path
    皆缺的行计入 bytes_without_hops（宁缺勿造）。

    返回是否消费到 relevant 路由行：True 时调用方跳过该记录的 legacy
    字段处理（history_transfer_bytes 等），防止同字节双计。holder 字段
    名若与 B3 调度器实际决策行有出入，待 B4b 对齐（B4b 有权修正）。
    """
    holders: list[list] = []
    if isinstance(decision, dict):
        top = decision.get("routes")
        if isinstance(top, list):
            holders.append(top)
        for key in RELEVANT_ROUTE_HOLDERS:
            nested = decision.get(key)
            if isinstance(nested, list):
                routes = nested
            else:
                routes = (nested.get("routes")
                          if isinstance(nested, dict) else None)
            if isinstance(routes, list):
                holders.append(routes)
    consumed = False
    for routes in holders:
        for row in routes:
            if not isinstance(row, dict):
                continue
            if row.get("category") not in RELEVANT_ROUTE_CATEGORIES:
                continue
            nbytes = row.get("bytes")
            if not isinstance(nbytes, int) or nbytes <= 0:
                continue
            consumed = True
            hops = _route_hops(row)
            rid = row.get("request_id") or request_id
            slot = per_request.setdefault(
                rid, {"actions": 0, "hop_bytes": 0,
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
    return consumed


def collect_wscllm(record: dict, acc: dict, per_request: dict) -> None:
    decision = record.get("decision") or {}
    request_id = record.get("request_id") or NA
    # relevant_distributed 变体（B4a/2026-09-02）：先扫 1000/3100/3300
    # 路由账；消费到路由行的记录不再走 legacy 字段（防双计）。
    if _accumulate_relevant_routes(decision, request_id, acc, per_request):
        return
    if record.get("kind") != "decode":
        # 历史迁移（prefill 决策）：2026-08-26 B2wp9py 起附带 noc_hops
        # 字段（实例图最短路）→ 纳入覆盖；旧产物无该字段时按
        # bytes_without_hops 处理（向后兼容）。
        if record.get("kind") == "prefill":
            nbytes = decision.get("history_transfer_bytes")
            if isinstance(nbytes, int) and nbytes > 0:
                hops = decision.get("noc_hops")
                slot = per_request.setdefault(
                    request_id, {"actions": 0, "hop_bytes": 0,
                                 "bytes_with": 0, "bytes_without": 0})
                if isinstance(hops, int) and hops >= 0:
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
    transfer = decision.get("prefill_decode_transfer")
    static_route = decision.get("static_route")
    nbytes = (transfer or {}).get("total_bytes") if isinstance(
        transfer, dict) else None
    hops = (static_route or {}).get("hop_count") if isinstance(
        static_route, dict) else None
    if not isinstance(nbytes, int) or nbytes <= 0:
        return
    slot = per_request.setdefault(
        request_id, {"actions": 0, "hop_bytes": 0,
                     "bytes_with": 0, "bytes_without": 0})
    if isinstance(hops, int) and hops >= 0:
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
        nbytes = decision.get("history_transfer_bytes")
        if not isinstance(nbytes, int):
            transfer = decision.get("history_transfer")
            if isinstance(transfer, dict):
                total = transfer.get("total_bytes")
                if isinstance(total, int):
                    nbytes = total
    elif kind == "decode":
        nbytes = decision.get("prefill_decode_transfer_bytes")
        if not isinstance(nbytes, int):
            transfer = decision.get("prefill_decode_transfer")
            if isinstance(transfer, dict):
                total = transfer.get("total_bytes")
                if isinstance(total, int):
                    nbytes = total
    if isinstance(nbytes, int) and nbytes > 0:
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
    旧产物（字段缺失）回退 collect_aggregate_only 语义（bytes 全部
    计入无覆盖侧），新旧产物可区分。"""
    decision = record.get("decision") or {}
    request_id = record.get("request_id") or NA
    kind = record.get("kind")
    relevant_kinds = ("prefill", "decode", "completion")
    hop_rows = decision.get("transfer_hop_bytes")
    if kind in relevant_kinds and isinstance(hop_rows, list):
        for row in hop_rows:
            if not isinstance(row, dict):
                continue
            nbytes = row.get("total_bytes")
            hop_bytes = row.get("noc_hop_bytes")
            shard_count = row.get("shard_count")
            if (not isinstance(nbytes, int) or nbytes <= 0
                    or not isinstance(shard_count, int) or shard_count <= 0):
                continue  # local_hit / 名义规模：无物理搬运
            slot = per_request.setdefault(
                request_id, {"actions": 0, "hop_bytes": 0,
                             "bytes_with": 0, "bytes_without": 0})
            slot["actions"] += 1
            slot["bytes_with"] += nbytes
            acc["actions_with_hops"] += 1
            acc["bytes_with_hops"] += nbytes
            if isinstance(hop_bytes, int) and hop_bytes > 0:
                slot["hop_bytes"] += hop_bytes
                acc["hop_bytes_total"] += hop_bytes
        return
    collect_aggregate_only(record, acc, per_request)


def collect_sh30(record: dict, acc: dict, per_request: dict) -> None:
    """S3（B4 升级，2026-08-26）：消费 decision log 的 shards[].noc_hops/
    noc_path 新字段（B3 60s 证据：completion_evictions 96 行 / 960 shards
    全带 noc_hops；ad-hoc 复核 hop_bytes=2383397232640）。

    count/fallback 语义与 S1 分支一致（_accumulate_shard_transfers：显式
    noc_hops 优先、noc_path 推导兜底、无路由 bytes 计 bytes_without_hops）。
    holder 集照 S1（history_transfer / prefill_decode_transfer / 各类
    逐出列表）；S3 现仅 completion_evictions 落 shards，其余 holder 留作
    字段同步后的前向兼容。B3 之前的 aggregate-only 停用口径
    （history_transfer_bytes / prefill_decode_transfer_bytes 无路由可聚）
    不再计入本账——那些记录无 noc 路由信息，不属于 Hop-Bytes 可判定
    宇宙；其量级（60s 参考跑 1045595684864 B）由门 JSON 另行登记。
    """
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


REPO_HOP_SOURCES: dict[str, dict] = {
    "astra-sim-sh_1.0": {
        "collector": collect_sh10,
        "granularity": "shard(noc_path)",
        "notes": "shards[].noc_path/noc_hops 已落盘 decision log",
    },
    "astra-sim-wscllm": {
        "collector": collect_wscllm,
        "granularity": "instance(static_route.hop_count)"
                       "+route(noc_hops:1000/3100/3300)",
        "notes": "legacy/session_lru：static_route 仅随 decode 决策落盘"
                 "（实例级粒度）；历史迁移无路由计入 bytes_without_hops。"
                 "relevant_distributed（B4a/2026-09-02）：kv_scatter/"
                 "kv_remote_reads/history_pull 类行携带生成侧 routes"
                 "（B2 发射函数返回结构，逐行 bytes+noc_hops，实例级"
                 "口径）；无 hops 行计 bytes_without_hops，不臆造",
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
           "bytes_with_hops": 0, "bytes_without_hops": 0}
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
