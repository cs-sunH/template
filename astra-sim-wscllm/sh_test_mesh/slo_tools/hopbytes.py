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
  astra-sim-face    决策日志未落任何 noc 路由字段（生成侧
                    generate_face_trace.py:474-506 的 routes 只进 trace
                    物化，不进 decision log）→ 无可聚合记录，
                    bytes 全部计入 bytes_without_hops（TODO_FACE：需
                    native 侧补落字段后重评）。
  astra-sim-sh_2.0  决策只落聚合 bytes（history_transfer_bytes 等），无
                    路由 → 同上（TODO_S2）。
  astra-sim-sh_3.0  同 S2（TODO_S3；sh30 决策只落聚合 bytes 与 affinity
                    reason）。

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
    """FACE/S2/S3：产物无 noc hops 字段——bytes 全部计入无覆盖侧。"""
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
        "collector": collect_aggregate_only,
        "granularity": "none",
        "notes": "TODO_FACE：decision log 未落 noc 路由字段（生成侧 "
                 "routes 仅进 trace 物化）——coverage=0，不臆造 hop 数",
    },
    "astra-sim-sh_2.0": {
        "collector": collect_aggregate_only,
        "granularity": "none",
        "notes": "TODO_S2：decision log 仅聚合 bytes，无路由——coverage=0",
    },
    "astra-sim-sh_3.0": {
        "collector": collect_aggregate_only,
        "granularity": "none",
        "notes": "TODO_S3：decision log 仅聚合 bytes，无路由——coverage=0",
    },
}


def cmd_hopbytes(args: argparse.Namespace) -> int:
    repo_variant = detect_repo_variant(args.run_dir, args.repo_variant)
    source = REPO_HOP_SOURCES.get(repo_variant)
    if source is None:
        fail(f"未登记的 repo_variant：{repo_variant}（REPO_HOP_SOURCES 需扩表）")
    acc = {"actions_with_hops": 0, "hop_bytes_total": 0,
           "bytes_with_hops": 0, "bytes_without_hops": 0}
    per_request: dict[str, dict] = {}
    for record in iter_jsonl(args.run_dir / DECISION_LOG_RELPATH):
        source["collector"](record, acc, per_request)
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
