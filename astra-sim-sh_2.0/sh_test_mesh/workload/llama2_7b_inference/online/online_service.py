#!/usr/bin/env python3
"""online_service.py -- sh_2.0 在线决策服务入口（方案 §4 步骤 1-8 操作 5）。

CLI：
    python3 online/online_service.py --bridge-dir <dir> --mode replay \
        --decision-log <path> --plan-dir <离线 manifest 目录>

流程：
  1. 载入 plan-dir/manifest.json（replay 权威之一）与 trace_config
     （load_face_trace_config；--config 可覆盖）；
  2. replay 模式：ReplaySource(decision_log) + GraphBatchBuilder +
     Sh20ReplayScheduler；manifest["requests"] 由 decision_log 的
     prefill/decode/completion 三流按 request 合并构造；
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
from online.replay_source import ReplaySource  # noqa: E402
from online.sh20_online_scheduler import Sh20OnlineScheduler  # noqa: E402
from online.sh20_replay_scheduler import Sh20ReplayScheduler  # noqa: E402


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


def _manifest_from_decision_log(decision_log_path: str, config) -> dict:
    """decision_log 三流按 request 合并 → manifest["requests"] 记录。

    合并键 = prefill 决策的标识/账本字段 + decode/completion 的图发射字段
    （graph_batch_builder 消费面）。"""
    prefill = {}
    decode = {}
    completion = {}
    with open(decision_log_path, "r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            kind = record["kind"]
            if kind == "prefill":
                prefill[record["request_id"]] = record
            elif kind == "decode":
                decode[record["request_id"]] = record
            elif kind == "completion":
                completion[record["request_id"]] = record
    requests = []
    for request_id, record in prefill.items():
        decision = record["decision"]
        merged = dict(decision)
        merged["request_id"] = request_id
        merged["prefill_record_tick"] = record["tick"]
        dec = decode[request_id]["decision"]
        for key in ("decode_instance_index", "prefill_decode_transfer",
                    "decode_evictions"):
            merged[key] = dec.get(key)
        comp = completion[request_id]["decision"]
        for key in ("completion_evictions", "kv_location_after_completion",
                    "kv_instance_after_completion"):
            merged[key] = comp.get(key)
        merged["completion_record_tick"] = completion[request_id]["tick"]
        merged["decode_record_tick"] = decode[request_id]["tick"]
        # 标识字段（prefill_length/decode_length/interval 等）从 config 队列
        # 补齐（决策日志的 prefill 决策不含重复的输入事实）。
        spec = config.request_queue[merged["queue_index"]]
        for key in ("prefill_length", "decode_length", "prefix_tokens",
                    "input_tokens_total", "session_arrival_time_ns",
                    "inter_request_interval_ns"):
            merged.setdefault(key, getattr(spec, key, None))
        requests.append(merged)
    requests.sort(key=lambda item: item["queue_index"])
    return {"requests": requests}


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
    parser.add_argument("--mode", default="replay", choices=("replay", "strategy"))
    parser.add_argument("--decision-log", default=None,
                        help="replay 模式：离线 decision_log.jsonl 路径")
    parser.add_argument("--plan-dir", required=True,
                        help="离线 manifest 目录（含 manifest.json；replay 模式"
                             "仍以 --decision-log 合并流为权威输入）")
    parser.add_argument("--config", default=None,
                        help="trace_config.csv 路径（缺省用 workload 默认）")
    parser.add_argument("--dump-nodes", action="store_true", default=False,
                        help="运行结束把构图器 per-rank 节点结构（id/name/"
                             "type/request_id/stage/tag/tensor_size/num_ops/"
                             "runtime_ns + 边表）流式写入 bridge 目录 "
                             "online_nodes.jsonl——B3 逐节点 canonical 比较与"
                             "发射序审计材料；中间产物，默认关")
    parser.add_argument("--sensing", action="store_true", default=False,
                        help="阶段 3 感知开关（默认关）：分层账本最小子集 + "
                             "两层剩余负载查询；查询/审计输入，不进策略判据")
    args = parser.parse_args(argv)

    if args.mode == "replay" and not args.decision_log:
        parser.error("replay 模式需要 --decision-log")
    if args.sensing and args.mode != "strategy":
        parser.error("--sensing 仅支持 --mode strategy（阶段 3 感知范围）")

    # plan-dir 存在性校验（fail-closed；replay 的 request 事实来自
    # decision_log 合并流，manifest.json 仅作存在性凭据）。
    if not os.path.isfile(os.path.join(args.plan_dir, "manifest.json")):
        parser.error(f"plan-dir 缺少 manifest.json: {args.plan_dir}")

    if args.config:
        config = load_face_trace_config(Path(args.config))
    else:
        config = load_face_trace_config()

    graph = GraphBatchBuilder(config, replay_clock=(args.mode == "replay"))
    digest_sink = _DigestSink(os.path.join(args.bridge_dir, DIGEST_LOG_NAME))
    if args.mode == "replay":
        replay = ReplaySource(args.decision_log)
        manifest = _manifest_from_decision_log(args.decision_log, config)
        scheduler = Sh20ReplayScheduler(
            manifest=manifest,
            config=config,
            replay=replay,
            graph=graph,
            digest_sink=digest_sink,
            mode=args.mode,
        )
    else:
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

    # B3/审计材料（阶段 2）：per-rank 节点 + 边，运行结束流式写出
    # （328,976 节点 ~35MB，有界：输入规模即上限；--dump-nodes 或
    # SH20_DUMP_NODES=1 开启，默认关）。
    if args.dump_nodes or os.environ.get("SH20_DUMP_NODES") == "1":
        with open(os.path.join(args.bridge_dir, "online_nodes.jsonl"),
                  "w", encoding="utf-8") as out:
            for rank, builder in sorted(graph.builders.items()):
                for node in builder.nodes:
                    out.write(json.dumps({
                        "rank": rank, "id": node["id"], "name": node["name"],
                        "type": node["type"],
                        "request_id": node.get("request_id", ""),
                        "stage": node.get("stage", ""),
                        "is_timer_op": node.get("is_timer_op", False),
                        "is_local_hbm_kv_restore": node.get(
                            "is_local_hbm_kv_restore", False),
                        "comm_tag": (
                            node["coll"]["priority"]
                            if node["type"] == 7 else node["comm"]["tag"]),
                        "comm_bytes": (
                            node["coll"]["bytes"]
                            if node["type"] == 7 else node["comm"]["bytes"]),
                        "tensor_size": node["compute"]["tensor_size"],
                        "num_ops": node["compute"]["num_ops"],
                        "runtime_ns": node["compute"]["runtime_ns"],
                    }, sort_keys=True) + "\n")
                for edge in builder.edges:
                    out.write(json.dumps({
                        "rank": rank, "edge_from": edge["from"],
                        "edge_to": edge["to"],
                    }, sort_keys=True) + "\n")

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

    # replay 模式 B1 口径出口：在线决策日志按权威 (record_tick, priority,
    # record_seq) 排序——与离线 decision_log 的 (tick, priority, seq) 同键
    # 同内容同序（触发序与记录序的解耦见 sh20_replay_scheduler 模块注释：
    # sh_2.0 的秒级 turn-0 准入排队使到达/准入边界天然分离）。strategy 模式
    # 行序 = 触发序（无权威日志）。
    log_rows = list(scheduler.online_log_rows)
    if args.mode == "replay":
        log_rows.sort(key=lambda row: (row.get("record_tick", row["tick"]),
                                       row["priority"],
                                       row.get("record_seq") if row.get("record_seq") is not None else row["seq"]))
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
