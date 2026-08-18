#!/usr/bin/env python3
"""online_service.py -- sh_2.0 在线决策服务入口（方案 §4 步骤 1-8 操作 5）。

CLI：
    python3 online/online_service.py --bridge-dir <dir> --mode strategy \
        --plan-dir <manifest 目录>

流程：
  1. 载入 trace_config（load_face_trace_config；--config 可覆盖）；
  2. strategy 模式：GraphBatchBuilder + Sh20OnlineScheduler；
     manifest["requests"] 由 trace_config 队列派生；
  3. BridgeServer.serve_forever（阻塞读 req_notify.fifo，零 polling；
     决策异常 => error response + exit 1，fail-closed）；
  4. EOF（C++ 关闭写端）= 运行结束：verify_run_end + 日志消费完整性校验，
     写 graph_batch_digests.jsonl / online_decision_log.jsonl 到 bridge_dir。

strategy 模式（步骤 1-9 的 Sh20OnlineScheduler）：真实策略（关感知）在
在线骨架中运行；manifest["requests"] 由 config.request_queue 经
_to_scheduler_requests + _validate_and_expand_requests（离线同源推导）构造。
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

from face_scheduler import PREFILL_CHUNK_SIZE  # noqa: E402
from generate_face_trace import (  # noqa: E402
    _to_scheduler_requests,
    load_face_trace_config,
)
from online.decision_bridge import BridgeServer  # noqa: E402
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402
from online.sh20_online_scheduler import Sh20OnlineScheduler  # noqa: E402


DIGEST_LOG_NAME = "graph_batch_digests.jsonl"
DECISION_LOG_NAME = "online_decision_log.jsonl"


class _DigestSink:
    """把每次 GraphBatch 产出的 digest 行顺带追加写入 bridge_dir。"""

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


def _manifest_from_config(config) -> dict:
    """strategy 模式：config.request_queue 经离线同源推导构造 manifest。"""
    from face_scheduler import _validate_and_expand_requests  # noqa: E402
    requests = _to_scheduler_requests(config.request_queue)
    runtimes, _ = _validate_and_expand_requests(requests, PREFILL_CHUNK_SIZE)
    records = []
    for runtime in runtimes:
        records.append({
            "queue_index": runtime.request.queue_index,
            "session_id": runtime.request.session_id,
            "turn_index": runtime.request.turn_index,
            "request_id": runtime.request.request_id,
            "prefill_length": runtime.request.prefill_length,
            "decode_length": runtime.request.decode_length,
            "prefix_tokens": runtime.request.prefix_tokens,
            "input_tokens_total": runtime.request.input_tokens_total,
            "session_arrival_time_ns": runtime.request.session_arrival_time_ns,
            "inter_request_interval_ns": (
                runtime.request.inter_request_interval_ns),
            "history_tokens_before": runtime.history_tokens_before,
            "prefill_tokens_to_process": runtime.prefill_tokens_to_process,
            "prefill_context_tokens": runtime.prefill_context_tokens,
            "final_context_tokens": runtime.final_context_tokens,
            "remaining_chunks": max(
                0, -(-runtime.prefill_tokens_to_process // PREFILL_CHUNK_SIZE)),
        })
    return {"requests": records}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="sh_2.0 online decision service")
    parser.add_argument("--bridge-dir", required=True)
    parser.add_argument("--mode", default="strategy", choices=("strategy",))
    parser.add_argument("--plan-dir", required=True,
                        help="manifest 目录（含 manifest.json；存在性凭据）")
    parser.add_argument("--config", default=None,
                        help="trace_config.csv 路径（缺省用 workload 默认）")
    parser.add_argument("--sensing", action="store_true", default=False,
                        help="阶段 3 感知开关（默认关）：分层账本最小子集 + "
                             "两层剩余负载查询；查询/审计输入，不进策略判据")
    args = parser.parse_args(argv)

    # plan-dir 存在性校验（fail-closed；strategy 的 request 事实由
    # trace_config 队列派生，manifest.json 仅作存在性凭据）。
    if not os.path.isfile(os.path.join(args.plan_dir, "manifest.json")):
        parser.error(f"plan-dir 缺少 manifest.json: {args.plan_dir}")

    if args.config:
        config = load_face_trace_config(Path(args.config))
    else:
        config = load_face_trace_config()

    graph = GraphBatchBuilder(config)
    digest_sink = _DigestSink(os.path.join(args.bridge_dir, DIGEST_LOG_NAME))
    manifest = _manifest_from_config(config)
    scheduler = Sh20OnlineScheduler(
        manifest=manifest,
        config=config,
        graph=graph,
        digest_sink=digest_sink,
        mode=args.mode,
        sensing=args.sensing,
    )

    server = BridgeServer(args.bridge_dir)
    result = server.serve_forever(
        scheduler.on_decision_batch,
        on_commit_ack=scheduler.on_commit_ack,
    )
    if result != 0:
        raise RuntimeError("serve_forever returned {}".format(result))

    scheduler.verify_run_end()
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

    # strategy 模式行序 = 触发序（无权威日志重排）。
    log_rows = list(scheduler.online_log_rows)
    _write_jsonl(os.path.join(args.bridge_dir, DECISION_LOG_NAME), log_rows)
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
