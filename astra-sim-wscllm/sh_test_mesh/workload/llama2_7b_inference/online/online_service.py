#!/usr/bin/env python3
"""online_service.py -- 在线决策服务入口(方案 §4 步骤 1-8 操作 5)。

CLI:
    python3 online/online_service.py --bridge-dir <dir> --mode strategy \
        --plan-dir <plan/manifest 目录>

流程:
  1. 载入 plan-dir/manifest.json(每 request 全部事实)与
     trace_config(load_wsc_llm_trace_config;--config 可覆盖);
  2. strategy 模式:GraphBatchBuilder + WscLlm( Legacy)OnlineScheduler;
  3. BridgeServer.serve_forever(scheduler.on_decision_batch,
     on_commit_ack=scheduler.on_commit_ack)——阻塞读 req_notify.fifo,
     零 polling;决策异常 => error response + exit 1(fail-closed);
  4. EOF(C++ 关闭写端)= 运行结束:verify_run_end + 日志消费完整性校验,
     并把 graph_batch_digests.jsonl / online_decision_log.jsonl 写入
     bridge_dir(每次 GraphBatch 产出顺带写 digest 行;决策行同批追加)。

strategy 模式(步骤 1-9 的 WscLlmOnlineScheduler):真实策略(关感知)在
在线骨架中运行,决策日志逐批写 online_decision_log.jsonl(决策由策略实时产出)。
"""

import argparse
import json
import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# import 路径:本文件位于 workload/llama2_7b_inference/online/,共享配置与
# 发射模块在上一级。路径只做 import 用途(红线:generate_wsc_llm_trace.py /
# wsc_llm_scheduler.py 只读 import 与注释)。
# --------------------------------------------------------------------------
_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _path in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from generate_wsc_llm_trace import load_wsc_llm_trace_config  # noqa: E402
from online.decision_bridge import BridgeServer  # noqa: E402
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402
from online.wsc_llm_legacy_online_scheduler import (  # noqa: E402
    WscLlmLegacyOnlineScheduler,
)
from online.wsc_llm_online_scheduler import WscLlmOnlineScheduler  # noqa: E402


DIGEST_LOG_NAME = "graph_batch_digests.jsonl"
DECISION_LOG_NAME = "online_decision_log.jsonl"


class _DigestSink:
    """把每次 GraphBatch 产出的 digest 行顺带追加写入 bridge_dir。"""

    def __init__(self, path: str):
        self.path = path
        self.count = 0
        # 覆盖旧文件(上一轮运行的残留)。
        with open(path, "w", encoding="utf-8"):
            pass

    def __call__(self, row: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as output:
            output.write(json.dumps(row, sort_keys=True) + "\n")
            # 阶段 7 §10.3:强制 flush 治理——digest 是流式审计行,崩溃/
            # 被杀时不得丢失尾部(open("a") 默认块缓冲,重定向下会滞留
            # ~8KB)。每行一次 flush 的开销对 3531 行可忽略。
            output.flush()
        self.count += 1


def _write_jsonl(path: str, rows: list) -> None:
    with open(path, "w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="wscllm online decision service")
    parser.add_argument("--bridge-dir", required=True,
                        help="bridge FIFO/request-response 目录(与 C++ 共享)")
    parser.add_argument("--mode", default="strategy", choices=("strategy",),
                        help="调度模式:strategy(唯一保留模式)")
    parser.add_argument("--plan-dir", required=True,
                        help="离线 plan/manifest 目录(含 manifest.json)")
    parser.add_argument("--config", default=None,
                        help="trace_config.csv 路径(缺省用 workload 默认)")
    parser.add_argument("--sensing", action="store_true", default=False,
                        help="阶段 3 感知开关(默认关,阶段 6 前默认关):分层"
                             "账本最小子集 + 两层剩余负载查询;感知数据是"
                             "查询/审计输入,不进策略判据,决策序列不变")
    args = parser.parse_args(argv)

    manifest_path = os.path.join(args.plan_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as source:
        manifest = json.load(source)

    # 缺省用 workload 默认 trace_config.csv(load_wsc_llm_trace_config 的
    # 默认参数);args.config 为 None 时不得传入(参数类型是 Path,None 会
    # 在 .exists() 处崩溃)。
    if args.config:
        # args.config 是 str;loader 签名是 Path(默认参数 CONFIG_CSV_PATH 亦
        # 为 Path),裸传 str 会在 .exists() 处崩溃——统一转 Path。
        config = load_wsc_llm_trace_config(Path(args.config))
    else:
        config = load_wsc_llm_trace_config()

    # 根因 #5(主控裁决 2026-08-15 第三项):strategy 模式保持物理链。
    graph = GraphBatchBuilder(config)
    digest_sink = _DigestSink(os.path.join(args.bridge_dir, DIGEST_LOG_NAME))
    # 步骤 1-9:真实策略(关感知,默认);决策日志逐批写 online_decision_log.jsonl。
    # 阶段 3:--sensing 开启感知(分层账本 + 两层剩余负载查询;查询/审计
    # 输入,不进策略判据,决策序列与关感知逐字节一致)。
    # 阶段 7 §10.6:strategy 模式按 config.kv_cache_policy 分发——主变体
    # session_lru_recompute -> WscLlmOnlineScheduler;第二变体 legacy ->
    # WscLlmLegacyOnlineScheduler(WSC Relevant(P,D) 静态域 + FCFS 队头
    # 阻塞)。分发不依赖任何代码默认值(总改造计划 §9.4:runner 显式
    # 传 kv_cache_policy;构造器各自 fail-closed 校验)。
    if config.kv_cache_policy == "session_lru_recompute":
        scheduler = WscLlmOnlineScheduler(
            manifest=manifest,
            config=config,
            graph=graph,
            digest_sink=digest_sink,
            mode=args.mode,
            sensing=args.sensing,
        )
    elif config.kv_cache_policy == "legacy":
        scheduler = WscLlmLegacyOnlineScheduler(
            manifest=manifest,
            config=config,
            graph=graph,
            digest_sink=digest_sink,
            mode=args.mode,
            sensing=args.sensing,
        )
    else:
        raise ValueError(
            "strategy 模式不支持 kv_cache_policy {!r}".format(
                config.kv_cache_policy))

    server = BridgeServer(args.bridge_dir)
    result = server.serve_forever(
        scheduler.on_decision_batch,
        on_commit_ack=scheduler.on_commit_ack,
    )
    if result != 0:
        raise RuntimeError("serve_forever returned {}".format(result))

    # 运行结束校验(fail-closed):
    scheduler.verify_run_end()
    # 阶段 7 §10.6:legacy 变体 run-end 终值(allocator 的 Relevant(P,D) 剩余
    # 容量;legacy 无逐事件 KV 日志,与 metrics_integration.kv_event_payload_
    # legacy 同构口径,run-end 终值审计件)。
    # 多态取用(仅 legacy 调度器提供;session_lru 无此产物)。
    legacy_payload = getattr(scheduler, "kv_event_payload_legacy", None)
    if callable(legacy_payload):
        with open(os.path.join(args.bridge_dir, "kv_event_payload_legacy.json"),
                  "w", encoding="utf-8") as output:
            json.dump(legacy_payload(), output, sort_keys=True)
            output.flush()
    # 阶段 4 §7.3:每决策批扫描条目数 profile(验收:与总 request 数无关,
    # full_scan_entries 恒为 0)。
    scheduler.dump_profile(os.path.join(args.bridge_dir, "profile.jsonl"))

    # 阶段 6 §9.1:online_stats.jsonl -- 分项计数器统一采集(Python 侧)。
    # 每已应用交付一行(合并桥接层每 request 服务时间) + 一行汇总。
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
        # callback 总数 = delivery_count(每次 on_decision_batch 进入 = 一次
        # Python callback;独立进程架构下无 engine-idle callback)。
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
        # 阶段 3:分层账本导出 + 决策边界两层剩余负载查询日志(结束总账核对
        # 与差异报告输入)。
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
