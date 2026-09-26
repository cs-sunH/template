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
      ⇒ 与逐 shard bytes×hops 求和严格相等，不依赖 shard bytes 分布；
      非同构 hops 一律 fail-closed，不静默算错；decode 的 kv_noc_hops
      与 shards 长度错位同样 fail-closed）。
    旧产物无字段 → bytes_without_hops（向后兼容，coverage 如实为低；
      prefill/decode 两分支同口径，decode 迁移字节不得静默丢出分母）。
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
                if len(set(hops)) != 1:
                    fail(f"history_noc_hops 跨 shard 不一致 {hops}"
                         f"——face 聚合口径要求同构 hops，fail-closed")
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
        if isinstance(shards, list) and isinstance(hops, list):
            if len(shards) != len(hops):
                fail(f"kv_noc_hops 与 shards 长度错位（{len(shards)} vs "
                     f"{len(hops)}）——face 口径要求一一对齐，fail-closed")
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
        else:
            # 旧产物无 kv_noc_hops（B2wp9py 前）→ 无 hops 可聚合：传输
            # bytes 按 total_bytes 全额计入 bytes_without_hops（coverage
            # 如实降低，不臆造 hop 数；与 prefill 分支回退口径一致）。
            # 不得把 decode P→D 迁移字节静默丢出分母。
            nbytes = (transfer or {}).get("total_bytes") if isinstance(
                transfer, dict) else None
            if isinstance(nbytes, int) and nbytes > 0:
                slot = _slot()
                slot["bytes_without"] += nbytes
                acc["bytes_without_hops"] += nbytes


def collect_wscllm(record: dict, acc: dict, per_request: dict) -> None:
    decision = record.get("decision") or {}
    request_id = record.get("request_id") or NA
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
        "granularity": "instance(static_route.hop_count)",
        "notes": "static_route 仅随 decode 决策落盘（实例级粒度）；"
                 "历史迁移无路由计入 bytes_without_hops",
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
