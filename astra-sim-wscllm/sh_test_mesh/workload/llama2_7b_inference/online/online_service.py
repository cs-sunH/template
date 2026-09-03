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
from metrics_integration import (  # noqa: E402
    MemoryActionRecorder,
    load_chiplets_per_npu,
)
from metrics_schema import MemoryMetricsObserver  # noqa: E402
from online.decision_bridge import BridgeServer  # noqa: E402
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402
from online.wsc_llm_legacy_online_scheduler import (  # noqa: E402
    WscLlmLegacyOnlineScheduler,
)
from online.wsc_llm_online_scheduler import WscLlmOnlineScheduler  # noqa: E402
from online.wsc_llm_relevant_online_scheduler import (  # noqa: E402
    WscLlmRelevantOnlineScheduler,
)
from session_kv_manager import set_metrics_observer  # noqa: E402


DIGEST_LOG_NAME = "graph_batch_digests.jsonl"
DECISION_LOG_NAME = "online_decision_log.jsonl"
TRAIN_LEDGER_LOG_NAME = "train_ledger.jsonl"
PROFILE_LOG_NAME = "profile.jsonl"
SENSING_QUERY_LOG_NAME = "sensing_query_log.jsonl"
ONLINE_STATS_LOG_NAME = "online_stats.jsonl"
# P1(2026-08-30)权威 HBM delta journal:SessionKVCacheManager 每次公开
# mutation 的逐 rank delta 流式落盘(append-only、行级 flush),run 末由
# verify_run_end 链上的 checksum 门流式重放对账(runner 归档进 results/)。
KV_DELTA_JOURNAL_NAME = "kv_delta_journal.jsonl"
# B3(2026-08-28):online_stats 基础行(未合并 processing_ns)的流式暂存
# 文件;结束期逐行读回、合并桥接层 processing_ns 后写出终文件并删除
# (终文件字节与改前结束一次性写出一致,行不再驻留内存)。
ONLINE_STATS_PARTIAL_NAME = "online_stats.jsonl.partial"


def _kv_delta_journal_enabled() -> bool:
    """P1 journal 开关:缺省 on;KV_DELTA_JOURNAL=off 完全旁路(不装
    recorder,不写 journal/checksum,行为与改前逐字节一致)。这是 journal
    唯一的开关,且缺省必须 on(权威账本与仿真同趟产出,零重跑成本)。"""

    return os.environ.get("KV_DELTA_JOURNAL", "on").strip().lower() not in {
        "0",
        "off",
        "false",
        "no",
    }


def _graph_digests_enabled() -> bool:
    """C2/D2(2026-08-28):graph_batch_digests.jsonl 开关,env SH_GRAPH_DIGESTS
    (默认 "0" 关)。digest 行是纯审计产物且每批全量二次序列化
    (nodes+edges 再 dumps+sha256),生产默认关;显式 =1 恢复改前恒写行为。
    (响应字节复用方案被否决:digest 需 sort_keys 口径而桥响应是另一序列
    化格式,复用会改桥协议字节——红线禁改。)"""
    return os.environ.get("SH_GRAPH_DIGESTS", "0") == "1"


class _JsonlSink:
    """jsonl 逐行落盘器(M3 流式落盘 + B2 常驻 fd,2026-08-23,批次B
    移植自 sh_3.0 母本)。

    常驻 fd:构造(或首行)时打开一次,__call__ 只 write+flush,结束
    close——不再每行 open/close。行格式与 _write_jsonl 完全一致
    (json.dumps(row, sort_keys=True) + "\\n"),行序不变 ⇒ 与改前结束
    一次性写出的文件逐字节相同。

    eager=True:构造即截断建文件(与改前"结束统一写出"对 0 行也建空
    文件的口径一致;digest/decision_log/profile 用);
    eager=False:首行才建文件(与改前"有行才写文件"的门语义一致,
    train_ledger 用)。
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
    (B2 后为 _JsonlSink 的常驻 fd 形态;语义/输出字节不变)。"""


class _NullSink:
    """D2(2026-08-28):丢弃型 sink——行计数保持、不序列化不落盘不驻留。
    供默认关的 jsonl(profile 等)使用:行 dict 由产出方构造(字段精简,
    开销可忽略),昂贵的 json.dumps+flush 与文件驻留全部跳过。"""

    def __init__(self):
        self.count = 0

    def __call__(self, row: dict) -> None:
        self.count += 1

    def close(self) -> None:
        pass


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
    # C2/D2(2026-08-28):digest 默认关(SH_GRAPH_DIGESTS=1 恢复恒写)。
    digest_sink = (
        _DigestSink(os.path.join(args.bridge_dir, DIGEST_LOG_NAME))
        if _graph_digests_enabled() else None)
    # M3 决策日志流式落盘(2026-08-23,批次B 移植自 sh_3.0 母本):
    # decision_log / train_ledger / profile 三类逐行 append+flush(行序与
    # json.dumps(sort_keys=True) 口径不变 ⇒ 输出文件与改前结束一次性
    # 写出逐字节相同),调度器不再驻留行列表(保留计数);崩溃/被杀时
    # 尾部不丢(§10.3 flush 治理)。train_ledger 用 eager=False(改前
    # "有行才写文件"的门语义)。
    # 本仓策略差异点(母本无):三路 sink 只接主变体
    # WscLlmOnlineScheduler;legacy 变体(非正式入口)不接 sink——仍走
    # 结束一次性写出(下方 legacy 兼容分支,输出字节与改前一致)。
    ledger_sink = None
    decision_log_sink = _JsonlSink(
        os.path.join(args.bridge_dir, DECISION_LOG_NAME))
    train_ledger_sink = _JsonlSink(
        os.path.join(args.bridge_dir, TRAIN_LEDGER_LOG_NAME), eager=False)
    profile_sink = (
        _JsonlSink(os.path.join(args.bridge_dir, PROFILE_LOG_NAME))
        if os.environ.get("SH_PROFILE_JSONL", "0") == "1" else _NullSink())
    # B3(2026-08-28):sensing_query / online_stats 流式落盘(行在产出时
    # 即完整;sensing_query 终文件字节与改前一致,online_stats 基础行
    # 进 .partial 暂存、结束期两遍合并)。legacy 变体不接(其行缓冲在
    # 调度器内,下方 getattr 门控的结束一次性写出兜底)。
    sensing_query_sink = _JsonlSink(
        os.path.join(args.bridge_dir, SENSING_QUERY_LOG_NAME),
        eager=args.sensing) if args.sensing else None
    online_stats_sink = _JsonlSink(
        os.path.join(args.bridge_dir, ONLINE_STATS_PARTIAL_NAME))
    # 步骤 1-9:真实策略(关感知,默认);决策日志逐批写 online_decision_log.jsonl。
    # 阶段 3:--sensing 开启感知(分层账本 + 两层剩余负载查询;查询/审计
    # 输入,不进策略判据,决策序列与关感知逐字节一致)。
    # 阶段 7 §10.6:strategy 模式按 config.kv_cache_policy 分发——主变体
    # session_lru_recompute -> WscLlmOnlineScheduler;第二变体 legacy ->
    # WscLlmLegacyOnlineScheduler(WSC Relevant(P,D) 静态域 + FCFS 队头
    # 阻塞);第三变体 relevant_distributed -> WscLlmRelevantOnlineScheduler
    # (D′ 分布式 KV 存放混合骨架,总文档 §2.4/裁决 #21,2026-09-02 B3)。
    # 分发不依赖任何代码默认值(总改造计划 §9.4:runner 显式传
    # kv_cache_policy;构造器各自 fail-closed 校验)。
    # P1 journal recorder:仅主变体且开关 on 时装配(见下方分支)。
    journal_recorder = None
    if config.kv_cache_policy == "session_lru_recompute":
        # P1(2026-08-30)权威 HBM delta journal 装配:仅主变体装 recorder
        # (legacy 变体无 SessionKVCacheManager,不适用);必须在调度器构造
        # 之前 install——manager 在构造期捕获全局 recorder 并落权重预载
        # 首 record。recorder 是纯观测旁路(不改任何决策);KV_DELTA_JOURNAL
        # =off 时不装,与改前行为完全一致。
        if _kv_delta_journal_enabled():
            journal_recorder = MemoryActionRecorder(
                MemoryMetricsObserver(load_chiplets_per_npu()),
                journal_path=os.path.join(
                    args.bridge_dir, KV_DELTA_JOURNAL_NAME),
            )
            set_metrics_observer(journal_recorder)
        # completed-unreconciled 行在 REQUEST_COMPLETE 边界立即写出；避免
        # 结束时由调度器聚合全量 completed ledger。
        ledger_sink = (
            _JsonlSink(os.path.join(args.bridge_dir, "ledger.jsonl"))
            if args.sensing else None)
        scheduler = WscLlmOnlineScheduler(
            manifest=manifest,
            config=config,
            graph=graph,
            digest_sink=digest_sink,
            decision_log_sink=decision_log_sink,
            train_ledger_sink=train_ledger_sink,
            profile_sink=profile_sink,
            mode=args.mode,
            sensing=args.sensing,
            sensing_query_sink=sensing_query_sink,
            online_stats_sink=online_stats_sink,
            ledger_sink=ledger_sink,
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
    elif config.kv_cache_policy == "relevant_distributed":
        # relevant_distributed 装配口径同 session_lru(全 sink;总文档
        # §2.4 最小侵入清单/执行文档 M5)。journal recorder:本变体无
        # SessionKVCacheManager,"构造器构造前 install" = 直接传入调度器
        # 构造参数(调度器转交 B1 分配器,权重预载行由分配器构造期落
        # journal;set_metrics_observer 的全局观察者仅被 manager 消费,
        # 本变体不装、不留死调用)。KV_DELTA_JOURNAL=off 时不装,行为
        # 与改前一致。
        if _kv_delta_journal_enabled():
            journal_recorder = MemoryActionRecorder(
                MemoryMetricsObserver(load_chiplets_per_npu()),
                journal_path=os.path.join(
                    args.bridge_dir, KV_DELTA_JOURNAL_NAME),
            )
        scheduler = WscLlmRelevantOnlineScheduler(
            manifest=manifest,
            config=config,
            graph=graph,
            digest_sink=digest_sink,
            decision_log_sink=decision_log_sink,
            train_ledger_sink=train_ledger_sink,
            profile_sink=profile_sink,
            mode=args.mode,
            sensing=args.sensing,
            online_stats_sink=online_stats_sink,
            ledger_sink=ledger_sink,
            kv_journal_recorder=journal_recorder,
        )
    else:
        raise ValueError(
            "strategy 模式不支持 kv_cache_policy {!r}".format(
                config.kv_cache_policy))
    # Scheduler 已提取构造期所需字段；不要让 main() 的局部变量再额外钉住
    # 完整 manifest。
    del manifest

    # Official peer is FileDecisionBridge: request files are exactly
    # nlohmann::json::dump() bytes (compact, no trailing newline).
    server = BridgeServer(
        args.bridge_dir, canonical_request_producer=True)
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
    # relevant_distributed 变体 run-end KV 事件载荷(16 元事件数组,
    # session_lru 同构;B1 分配器事件流三类 placement/release/history_pull,
    # 总文档裁决 #23)。多态取用(仅 relevant 调度器提供)。
    relevant_payload = getattr(scheduler, "kv_event_payload_relevant", None)
    if callable(relevant_payload):
        with open(
                os.path.join(args.bridge_dir,
                             "kv_event_payload_relevant.json"),
                "w", encoding="utf-8") as output:
            json.dump(relevant_payload(), output, sort_keys=True)
            output.flush()
    # 阶段 4 §7.3:每决策批扫描条目数 profile(验收:与总 request 数无关,
    # full_scan_entries 恒为 0)——主变体 M3 起已在 build_graph_batch 逐行
    # 流式写出,此处仅 legacy 变体(未接 sink)结束一次性写出。
    if scheduler.profile_sink is None:
        scheduler.dump_profile(os.path.join(args.bridge_dir, "profile.jsonl"))

    # 阶段 6 §9.1:online_stats.jsonl -- 分项计数器统一采集(Python 侧)。
    # 每已应用交付一行(合并桥接层每 request 服务时间) + 一行汇总。
    # B3(2026-08-28):主变体行已流式写入 .partial(未合并 processing_ns),
    # 此处逐行读回、合并、写终文件后删除暂存——行不再驻留内存,终文件
    # 字节与改前结束一次性写出一致;legacy 变体(未接 sink)仍走缓冲行。
    bridge_stats = server.stats()
    summary_row = {
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
    }
    stats_path = os.path.join(args.bridge_dir, ONLINE_STATS_LOG_NAME)
    partial_path = os.path.join(args.bridge_dir, ONLINE_STATS_PARTIAL_NAME)

    def _base_stat_rows():
        if getattr(scheduler, "online_stats_sink", None) is not None:
            with open(partial_path, "r", encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        yield json.loads(line)
        else:
            yield from scheduler.online_stats_rows

    def _merged_stat_rows():
        # 两个输入都是严格 delivery_sequence 顺序流,用 merge join 保持 O(1)
        # 内存。任何缺行、重行或乱序都 fail-closed,不能静默漏 processing_ns。
        processing_rows = iter(server.per_request_stats())
        processing = next(processing_rows, None)
        previous_seq = -1
        for base_row in _base_stat_rows():
            row = dict(base_row)
            seq = row["delivery_sequence"]
            if seq <= previous_seq:
                raise RuntimeError(
                    "online_stats delivery_sequence is not strictly increasing: "
                    "{} after {}".format(seq, previous_seq))
            if processing is None or processing["seq"] != seq:
                got = None if processing is None else processing["seq"]
                raise RuntimeError(
                    "online_stats/bridge processing stream mismatch: stats seq "
                    "{} vs processing seq {}".format(seq, got))
            row["processing_ns"] = processing["processing_ns"]
            yield row
            previous_seq = seq
            processing = next(processing_rows, None)
        if processing is not None:
            raise RuntimeError(
                "orphan bridge processing row for seq {}".format(
                    processing["seq"]))

    online_stats_sink.close()
    with open(stats_path, "w", encoding="utf-8") as output:
        for row in _merged_stat_rows():
            output.write(json.dumps(row, sort_keys=True) + "\n")
        output.write(json.dumps(summary_row, sort_keys=True) + "\n")
    if os.path.exists(partial_path):
        os.unlink(partial_path)
    server.discard_per_request_stats()

    # online_decision_log / train_ledger:主变体已由 _JsonlSink 逐行落盘
    # (M3);legacy 变体未接 sink,保留结束一次性写出(行缓冲兼容路径,
    # 输出与改前逐字节一致)。
    if scheduler.decision_log_sink is None:
        _write_jsonl(os.path.join(args.bridge_dir, DECISION_LOG_NAME),
                     scheduler.online_log_rows)
    # 拼 batch 列车台账(§7.3 不变量断言输入;每次列车/整段发射一行;
    # §3.6 PD 分离:D 侧行 = 迭代列车,P 侧行 = 退化纯 prefill 列车)。
    if (getattr(scheduler, "train_ledger_sink", None) is None
            and getattr(scheduler, "train_ledger_rows", None)):
        _write_jsonl(os.path.join(args.bridge_dir, TRAIN_LEDGER_LOG_NAME),
                     scheduler.train_ledger_rows)
    if args.sensing:
        # 阶段 3:分层账本导出(结束总账核对与差异报告输入;感知关闭的
        # 正式跑不经过这里)。B3:sensing_query_log 已流式落盘,不再结束
        # 一次性写出(legacy 变体不挂感知,fail-closed,不受影响)。
        scheduler.dump_ledger(os.path.join(args.bridge_dir, "ledger.jsonl"))
    # B2/M3/B3:常驻 fd 显式关闭(serve 期之外无写入;异常路径由进程退出
    # 兜底)。digest/profile 关闭态为 None/_NullSink,同样安全。legacy
    # 变体虽未消费行,构造时 eager 建立的空文件也在此关闭(其内容随后
    # 被上方结束一次性写出覆盖,字节口径不变)。
    for sink in (digest_sink, decision_log_sink, train_ledger_sink,
                 profile_sink, sensing_query_sink, ledger_sink):
        if sink is not None:
            sink.close()
    # P1 journal:verify_run_end 的 checksum 门(重放)已在上方完成,此处
    # 关闭常驻 journal fd;旁路态(None)与 legacy 分支不经过这里。
    if journal_recorder is not None:
        journal_recorder.close_journal()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- CLI 顶层 fail-closed
        print("online_service: fatal: {}: {}".format(type(exc).__name__, exc),
              file=sys.stderr)
        sys.exit(1)
