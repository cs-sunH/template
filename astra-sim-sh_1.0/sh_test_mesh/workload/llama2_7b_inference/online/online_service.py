#!/usr/bin/env python3
"""online_service.py -- sh_1.0 在线决策服务入口(方案 §4 步骤 1-8 操作 5)。

CLI(与步骤 1-8 验证命令一致):
    python3 online/online_service.py --bridge-dir <dir> --mode replay \
        --decision-log <path> --plan-dir <离线 plan/manifest 目录>

本仓单策略变体(无 kv_cache_policy 分发;方案 §4.1 第 1 条):
  - replay 模式:ReplaySource(decision_log)+ GraphBatchBuilder(replay_clock)
    + Sh10ReplayScheduler;
  - strategy 模式:Sh10OnlineScheduler(真实策略,关感知默认;--sensing 开
    感知,LUT 冻结表 face_lut.csv 按合同⑨加载)。
"""

import argparse
import json
import os
import sys
from pathlib import Path

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _path in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from generate_face_trace import load_face_trace_config  # noqa: E402
from online.decision_bridge import BridgeServer  # noqa: E402
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402
from online.replay_source import ReplaySource  # noqa: E402
from online.sh10_online_scheduler import Sh10OnlineScheduler  # noqa: E402
from online.sh10_replay_scheduler import Sh10ReplayScheduler  # noqa: E402


DIGEST_LOG_NAME = "graph_batch_digests.jsonl"
DECISION_LOG_NAME = "online_decision_log.jsonl"


class _DigestSink:
    """把每次 GraphBatch 产出的 digest 行顺带追加写入 bridge_dir(逐行 flush)。"""

    def __init__(self, path: str):
        self.path = path
        self.count = 0
        with open(path, "w", encoding="utf-8"):
            pass

    def __call__(self, row: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as output:
            output.write(json.dumps(row, sort_keys=True) + "\n")
            output.flush()
        self.count += 1


def _write_jsonl(path: str, rows: list) -> None:
    with open(path, "w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="sh_1.0 online decision service")
    parser.add_argument("--bridge-dir", required=True,
                        help="bridge FIFO/request-response 目录(与 C++ 共享)")
    parser.add_argument("--mode", default="replay", choices=("replay", "strategy"),
                        help="调度模式:replay(步骤 1-8);strategy(步骤 1-9)")
    parser.add_argument("--decision-log", default=None,
                        help="replay 模式:离线 decision_log.jsonl 路径")
    parser.add_argument("--plan-dir", required=True,
                        help="离线 plan/manifest 目录(含 manifest.json)")
    parser.add_argument("--config", default=None,
                        help="trace_config.csv 路径(缺省用 workload 默认)")
    parser.add_argument("--frozen-lut", default=None,
                        help="strategy 模式:冻结 LUT 表 face_lut.csv 路径"
                             "(缺省 = sh_test_mesh/generated/ 下唯一 ET 目录内的"
                             " face_lut.csv;裸仓库态先生成,见 traces/PROVENANCE.md)")
    parser.add_argument("--sensing", action="store_true", default=False,
                        help="阶段 3 感知开关(默认关):分层账本最小子集 + "
                             "两层剩余负载查询;感知数据是查询/审计输入,不进"
                             "策略判据,决策序列不变")
    args = parser.parse_args(argv)

    if args.mode == "replay" and not args.decision_log:
        parser.error("replay 模式需要 --decision-log")
    if args.sensing and args.mode != "strategy":
        parser.error("--sensing 仅支持 --mode strategy(阶段 3 感知范围)")

    manifest_path = os.path.join(args.plan_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as source:
        manifest = json.load(source)

    if args.config:
        config = load_face_trace_config(Path(args.config))
    else:
        config = load_face_trace_config()

    # 蓝本裁决 4③(三段推广):replay 模式段 1 清 prefill 组 previous_id
    # (LUT 时钟并发);strategy 模式保持物理链。
    graph = GraphBatchBuilder(config, replay_clock=(args.mode == "replay"))
    digest_sink = _DigestSink(os.path.join(args.bridge_dir, DIGEST_LOG_NAME))
    if args.mode == "replay":
        replay = ReplaySource(args.decision_log)
        scheduler = Sh10ReplayScheduler(
            manifest=manifest,
            config=config,
            replay=replay,
            graph=graph,
            digest_sink=digest_sink,
            mode=args.mode,
        )
    else:
        scheduler = Sh10OnlineScheduler(
            manifest=manifest,
            config=config,
            graph=graph,
            digest_sink=digest_sink,
            mode=args.mode,
            sensing=args.sensing,
            frozen_lut_csv=args.frozen_lut,
        )

    server = BridgeServer(args.bridge_dir)
    result = server.serve_forever(
        scheduler.on_decision_batch,
        on_commit_ack=scheduler.on_commit_ack,
    )
    if result != 0:
        raise RuntimeError("serve_forever returned {}".format(result))

    scheduler.verify_run_end()
    if args.mode == "replay" and not replay.consumed_all():
        raise RuntimeError(
            "run ended with unconsumed replay decisions: consumed={} total={}"
            .format(replay.consumed_counts(), replay.total_counts()))
    scheduler.dump_profile(os.path.join(args.bridge_dir, "profile.jsonl"))

    bridge_stats = server.stats()
    per_request = {row["seq"]: row for row in server.per_request_stats()}
    stats_rows = []
    for row in scheduler.online_stats_rows:
        merged = dict(row)
        seq_info = per_request.get(row["delivery_sequence"])
        if seq_info is not None:
            merged["processing_ns"] = seq_info["processing_ns"]
        stats_rows.append(merged)
    stats_rows.append({
        "summary": True,
        "mode": args.mode,
        "sensing": args.sensing,
        "delivery_count": scheduler.delivery_count,
        "ack_count": scheduler.ack_count,
        "python_callback_count_by_reason":
            scheduler.python_callback_count_by_reason,
        "python_callback_total": scheduler.delivery_count,
        "scheduler_self_ns_total": scheduler.scheduler_self_ns_total,
        "scheduler_self_ns_avg_per_delivery": (
            scheduler.scheduler_self_ns_total // scheduler.delivery_count
            if scheduler.delivery_count else 0),
        "gil_wait_ns": bridge_stats["gil_wait_ns"],
        "channel_bytes": bridge_stats["channel_bytes"],
        "forced_flush_count": bridge_stats["forced_flush_count"],
        "handler_calls": bridge_stats["handler_calls"],
    })
    _write_jsonl(os.path.join(args.bridge_dir, "online_stats.jsonl"),
                 stats_rows)

    _write_jsonl(os.path.join(args.bridge_dir, DECISION_LOG_NAME),
                 scheduler.online_log_rows)
    if args.sensing:
        scheduler.dump_ledger(os.path.join(args.bridge_dir, "ledger.jsonl"))
        _write_jsonl(
            os.path.join(args.bridge_dir, "sensing_query_log.jsonl"),
            scheduler.sensing_query_rows)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- CLI 顶层 fail-closed
        print("online_service: fatal: {}: {}".format(type(exc).__name__, exc),
              file=sys.stderr)
        sys.exit(1)
