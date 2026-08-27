#!/usr/bin/env python3
"""online_service.py -- sh_1.0 在线决策服务入口(方案 §4 步骤 1-8 操作 5)。

CLI:
    python3 online/online_service.py --bridge-dir <dir> --mode strategy \
        --plan-dir <plan/manifest 目录>

本仓单策略变体(无 kv_cache_policy 分发;方案 §4.1 第 1 条):
  - strategy 模式:Sh10OnlineScheduler(真实策略,关感知默认;--sensing 开
    感知,Decode 代价按当前状态直接计算 Roofline 估计)。
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
from online.sh10_online_scheduler import Sh10OnlineScheduler  # noqa: E402


DIGEST_LOG_NAME = "graph_batch_digests.jsonl"
DECISION_LOG_NAME = "online_decision_log.jsonl"
TRAIN_LEDGER_LOG_NAME = "train_ledger.jsonl"
PROFILE_LOG_NAME = "profile.jsonl"


class _JsonlSink:
    """jsonl 逐行落盘器（M3 流式落盘 + B2 常驻 fd，2026-08-23）。

    常驻 fd：构造（或首行）时打开一次，__call__ 只 write+flush，结束
    close——不再每行 open/close。行格式与 _write_jsonl 完全一致
    （json.dumps(row, sort_keys=True) + "\\n"），行序不变 ⇒ 与改前结束
    一次性写出的文件逐字节相同。

    eager=True：构造即截断建文件（与改前"结束统一写出"对 0 行也建空
    文件的口径一致；digest/decision_log/profile 用）；
    eager=False：首行才建文件（与改前"有行才写文件"的门语义一致，
    train_ledger 用）。
    """

    def __init__(self, path: str, *, eager: bool = True):
        self.path = path
        self.count = 0
        self._output = None
        if eager:
            self._open()

    def _open(self) -> None:
        # 覆盖旧文件(上一轮运行的残留)。
        self._output = open(self.path, "w", encoding="utf-8")

    def __call__(self, row: dict) -> None:
        if self._output is None:
            self._open()
        self._output.write(json.dumps(row, sort_keys=True) + "\n")
        # 阶段 7 §10.3:强制 flush 治理——流式审计行,崩溃/被杀时不得丢失
        # 尾部(常驻 fd 缓冲会滞留;每行一次 flush 的开销可忽略)。
        self._output.flush()
        self.count += 1

    def close(self) -> None:
        if self._output is not None:
            self._output.close()
            self._output = None


class _DigestSink(_JsonlSink):
    """把每次 GraphBatch 产出的 digest 行顺带追加写入 bridge_dir
    （B2 后为 _JsonlSink 的常驻 fd 形态；语义/输出字节不变）。"""


def _write_jsonl(path: str, rows: list) -> None:
    with open(path, "w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="sh_1.0 online decision service")
    parser.add_argument("--bridge-dir", required=True,
                        help="bridge FIFO/request-response 目录(与 C++ 共享)")
    parser.add_argument("--mode", default="strategy", choices=("strategy",),
                        help="调度模式:strategy(步骤 1-9;唯一保留模式)")
    parser.add_argument("--plan-dir", required=True,
                        help="plan/manifest 目录(含 manifest.json)")
    parser.add_argument("--config", default=None,
                        help="trace_config.csv 路径(缺省用 workload 默认)")
    parser.add_argument("--sensing", action="store_true", default=False,
                        help="阶段 3 感知开关(默认关):分层账本最小子集 + "
                             "两层剩余负载查询;感知数据是查询/审计输入,不进"
                             "策略判据,决策序列不变")
    args = parser.parse_args(argv)

    manifest_path = os.path.join(args.plan_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as source:
        manifest = json.load(source)

    if args.config:
        config = load_face_trace_config(Path(args.config))
    else:
        config = load_face_trace_config()

    # 蓝本裁决 4③(三段推广):strategy 模式保持物理链。
    graph = GraphBatchBuilder(config)
    digest_sink = _DigestSink(os.path.join(args.bridge_dir, DIGEST_LOG_NAME))
    # M3 决策日志流式落盘(2026-08-23):decision_log / train_ledger /
    # profile 三类逐行 append+flush(行序与 json.dumps(sort_keys=True)
    # 口径不变 ⇒ 输出文件与改前结束一次性写出逐字节相同),调度器不再
    # 驻留行列表(保留计数);崩溃/被杀时尾部不丢(§10.3 flush 治理)。
    # train_ledger 用 eager=False(改前"有行才写文件"的门语义)。
    decision_log_sink = _JsonlSink(
        os.path.join(args.bridge_dir, DECISION_LOG_NAME))
    train_ledger_sink = _JsonlSink(
        os.path.join(args.bridge_dir, TRAIN_LEDGER_LOG_NAME), eager=False)
    profile_sink = _JsonlSink(
        os.path.join(args.bridge_dir, PROFILE_LOG_NAME))
    scheduler = Sh10OnlineScheduler(
        manifest=manifest,
        config=config,
        graph=graph,
        digest_sink=digest_sink,
        decision_log_sink=decision_log_sink,
        train_ledger_sink=train_ledger_sink,
        profile_sink=profile_sink,
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
    # 阶段 4 §7.3:每决策批扫描条目数 profile(验收:与总 request 数无关,
    # full_scan_entries 恒为 0)——M3 起已在 build_graph_batch 逐行流式
    # 写出,此处不再结束一次性写出。

    # 阶段 6 §9.1:online_stats.jsonl -- 分项计数器统一采集(Python 侧)。
    # 每已应用交付一行(合并桥接层每 request 服务时间) + 一行汇总。
    # (M3 残留:每行需结束期合并桥接层 processing_ns,无字节等价的流式
    # 设计,保留结束一次性写出——行量小(每交付一行、字段精简)。)
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

    # online_decision_log / train_ledger 已由 _JsonlSink 逐行落盘(M3)。
    if args.sensing:
        # 阶段 3:分层账本导出 + 决策边界两层剩余负载查询日志(结束总账核对
        # 与差异报告输入;感知关闭的正式跑不经过这里)。
        scheduler.dump_ledger(os.path.join(args.bridge_dir, "ledger.jsonl"))
        _write_jsonl(
            os.path.join(args.bridge_dir, "sensing_query_log.jsonl"),
            scheduler.sensing_query_rows)
    # B2/M3:常驻 fd 显式关闭(serve 期之外无写入;异常路径由进程退出兜底)。
    for sink in (digest_sink, decision_log_sink, train_ledger_sink,
                 profile_sink):
        sink.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- CLI 顶层 fail-closed
        print("online_service: fatal: {}: {}".format(type(exc).__name__, exc),
              file=sys.stderr)
        sys.exit(1)
