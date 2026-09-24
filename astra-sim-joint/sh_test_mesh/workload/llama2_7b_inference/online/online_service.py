#!/usr/bin/env python3
"""online_service.py -- 在线决策服务入口(方案 §4 步骤 1-8 操作 5)。

CLI:
    python3 online/online_service.py --bridge-dir <dir> --mode strategy \
        --plan-dir <plan/manifest 目录>

流程:
  1. 载入 plan-dir/manifest.json(每 request 全部事实)与
     trace_config(load_face_trace_config;--config 可覆盖);
  2. strategy 模式:GraphBatchBuilder + Sh30OnlineScheduler;
  3. BridgeServer.serve_forever(scheduler.on_decision_batch,
     on_commit_ack=scheduler.on_commit_ack)——阻塞读 req_notify.fifo,
     零 polling;决策异常 => error response + exit 1(fail-closed);
  4. EOF(C++ 关闭写端)= 运行结束:verify_run_end + 日志消费完整性校验,
     并把 graph_batch_digests.jsonl / online_decision_log.jsonl 写入
     bridge_dir(每次 GraphBatch 产出顺带写 digest 行;决策行同批追加)。

strategy 模式(步骤 1-9 的 Sh30OnlineScheduler):真实策略(关感知)在
在线骨架中运行,决策日志逐批写 online_decision_log.jsonl(决策由策略实时产出)。
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

# --------------------------------------------------------------------------
# import 路径:本文件位于 workload/llama2_7b_inference/online/,离线写出模块
# 在上一级。路径只做 import 用途(红线:generate_face_trace.py /
# face_scheduler.py 只读 import 与注释)。
# --------------------------------------------------------------------------
_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _path in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from generate_face_trace import load_face_trace_config  # noqa: E402
from online.decision_bridge import BridgeServer  # noqa: E402
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402
from online.sh30_online_scheduler import Sh30OnlineScheduler  # noqa: E402


DIGEST_LOG_NAME = "graph_batch_digests.jsonl"
DECISION_LOG_NAME = "online_decision_log.jsonl"
TRAIN_LEDGER_LOG_NAME = "train_ledger.jsonl"
PROFILE_LOG_NAME = "profile.jsonl"
SENSING_QUERY_LOG_NAME = "sensing_query_log.jsonl"
ONLINE_STATS_LOG_NAME = "online_stats.jsonl"
# B3(2026-08-28):online_stats 基础行(未合并 processing_ns)的流式暂存
# 文件;结束期逐行读回、合并桥接层 processing_ns 后写出终文件并删除
# (终文件字节与改前结束一次性写出一致,行不再驻留内存)。
ONLINE_STATS_PARTIAL_NAME = "online_stats.jsonl.partial"


def _graph_digests_enabled() -> bool:
    """C2/D2(2026-08-28):graph_batch_digests.jsonl 开关,env SH_GRAPH_DIGESTS
    (默认 "0" 关)。digest 行是纯审计产物且每批全量二次序列化
    (nodes+edges 再 dumps+sha256),生产默认关;显式 =1 恢复改前恒写行为。
    (响应字节复用方案被否决:digest 需 sort_keys 口径而桥响应是另一序列
    化格式,复用会改桥协议字节——红线禁改。)"""
    return os.environ.get("SH_GRAPH_DIGESTS", "0") == "1"


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


def dump_joint_kv_ledgers(scheduler, path: str) -> None:
    """R4/K6 台账侧车落盘（merge_degrade + deep_gap）。

    终审-中4：由 main 的 try/finally 全路径调用（RED run 异常退出也落
    终态缺口现场——deep_gap 落账 = run 终止的 K6 语义）。四审-低8：
    原子写（tmp + os.replace）——满盘/SIGKILL 半截文件触发面消除。
    诊断通道尽力而为：落盘自身失败只打印栈、不得改写主路径退出语义。

    合并方向 v2（2026-09-17）：merge_degrade_events 旧机制（R4 分层
    自降级产出）已退役——台账冻结（旧侧车文件的兼容读取保留：键恒在、
    新 run 恒空列表），产出端在 kv_manager 侧删除。属性可能随退役被
    移除，getattr 缺省空序列防 AttributeError（deep_gap 通道不变）。

    C11 rider（C13 移交）：copy_handoff_events（C13 copy 逐 chunk 四步
    交接的事件级计量——科目 rid#handoff / rid#copy-stream，守恒式
    H_home + H_exec = H + D_handoff 的被检对象）随本侧车一行落盘；
    C13 交付期为 __new__ 替身/旧调用方兼容，getattr 缺省空序列。

    F3（C14 §20.6-2 / C16 §22.7-1 移交义务履行）：kv_delta_journal
    （C14 结算时刻逐请求事实——字段面 = _append_kv_delta_row 冻结口径）
    随本侧车第四键落盘；行源 = kv_delta_journal_rows()（C14 字段集/
    seq 链完整性 fail-closed 导出）。两文件同卡落地，无版本偏斜，
    直接经访问器取行。

    A12'（2026-09-22，§4.3 补遗）：逐键独立导出——单键失败写
    ``<key>_export_error`` 哨兵字符串（其余键不受连坐；kv_delta_
    journal 的 seq 断链 fail-closed 经哨兵显式落盘，消费端按最弱层
    + 明示注记处理），不再被兜底 except 吞成"四键尽失 + 消费端误判
    零结算"。
    """
    # A12'：逐键生产隔离。
    producers = (
        ("merge_degrade_events",
         lambda: [dict(event) for event in
                  getattr(scheduler.kv_manager,
                          "merge_degrade_events", ())]),
        ("deep_gap_events",
         lambda: [dict(event) for event in
                  scheduler.kv_manager.deep_gap_events]),
        ("copy_handoff_events",
         lambda: [dict(event) for event in
                  getattr(scheduler.kv_manager,
                          "copy_handoff_events", ())]),
        ("kv_delta_journal",
         lambda: [dict(row) for row in
                  scheduler.kv_manager.kv_delta_journal_rows()]),
    )
    payload: dict = {}
    for key, produce in producers:
        try:
            payload[key] = produce()
        except Exception as exc:  # noqa: BLE001 -- 单键哨兵，不阻断
            traceback.print_exc()
            payload[key + "_export_error"] = f"{type(exc).__name__}: {exc}"
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as sink:
            json.dump(payload, sink, indent=1, sort_keys=True)
        os.replace(tmp_path, path)
    except Exception:  # noqa: BLE001 -- 诊断通道不得阻断主异常路径
        traceback.print_exc()


def _update_joint_manifest_telemetry(scheduler, path: str) -> None:
    """C11 rider（C8-BLOCKED①）：遥测完备性五键回填 manifest 侧车。

    run 全程才能判定的 sticky 事实（键缺席/窗口破损/覆盖终值）在
    verify_run_end 之后经读-改-写原子更新（tmp + os.replace）落
    joint_mechanism_manifest.json——G2 门"collective_coverage 翻转条件
    落 manifest"的侧车半（决策日志收尾行 kind=link_telemetry_coverage
    为另一半，C8 已交付）。诊断通道尽力而为：失败只打印栈，不改写主
    路径退出语义。"""
    payload = {
        "telemetry_epoch_count": scheduler._link_telemetry_epoch_count,
        "telemetry_sample_count": scheduler._link_telemetry_sample_count,
        "telemetry_key_absent_seen": scheduler._telemetry_absent_seen,
        "telemetry_window_broken": scheduler._telemetry_window_broken,
        "collective_coverage": (
            scheduler._joint_flows.collective_coverage),
    }
    tmp_path = path + ".tmp"
    try:
        with open(path, "r", encoding="utf-8") as source:
            manifest = json.load(source)
        manifest["link_telemetry"] = payload
        with open(tmp_path, "w", encoding="utf-8") as sink:
            json.dump(manifest, sink, indent=1, sort_keys=True)
        os.replace(tmp_path, path)
    except Exception:  # noqa: BLE001 -- 诊断通道不得阻断主异常路径
        traceback.print_exc()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="sh_3.0 online decision service")
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

    # 缺省用 workload 默认 trace_config.csv(load_face_trace_config 的
    # 默认参数);args.config 为 None 时不得传入(参数类型是 Path,None 会
    # 在 .exists() 处崩溃)。
    if args.config:
        # args.config 是 str;loader 签名是 Path(默认参数 CONFIG_CSV_PATH 亦
        # 为 Path),裸传 str 会在 .exists() 处崩溃——统一转 Path。
        config = load_face_trace_config(Path(args.config))
    else:
        config = load_face_trace_config()

    # strategy 模式保持物理跨 request 链(根因 #5 裁决:物理链为③④口径)。
    graph = GraphBatchBuilder(config)
    # C2/D2(2026-08-28):digest 默认关(SH_GRAPH_DIGESTS=1 恢复恒写)。
    digest_sink = (
        _DigestSink(os.path.join(args.bridge_dir, DIGEST_LOG_NAME))
        if _graph_digests_enabled() else None)
    # M3 决策日志流式落盘(2026-08-23):decision_log / train_ledger /
    # profile 三类逐行 append+flush(行序与 json.dumps(sort_keys=True)
    # 口径不变 ⇒ 输出文件与改前结束一次性写出逐字节相同),调度器不再
    # 驻留行列表(保留计数);崩溃/被杀时尾部不丢(§10.3 flush 治理)。
    # train_ledger 用 eager=False(改前"有行才写文件"的门语义)。
    decision_log_sink = _JsonlSink(
        os.path.join(args.bridge_dir, DECISION_LOG_NAME))
    train_ledger_sink = _JsonlSink(
        os.path.join(args.bridge_dir, TRAIN_LEDGER_LOG_NAME), eager=False)
    profile_sink = (
        _JsonlSink(os.path.join(args.bridge_dir, PROFILE_LOG_NAME))
        if os.environ.get("SH_PROFILE_JSONL", "0") == "1" else _NullSink())
    # B3(2026-08-28):sensing_query / online_stats 流式落盘(行在产出时
    # 即完整;sensing_query 终文件字节与改前一致,online_stats 基础行
    # 进 .partial 暂存、结束期两遍合并)。
    sensing_query_sink = _JsonlSink(
        os.path.join(args.bridge_dir, SENSING_QUERY_LOG_NAME),
        eager=args.sensing) if args.sensing else None
    online_stats_sink = _JsonlSink(
        os.path.join(args.bridge_dir, ONLINE_STATS_PARTIAL_NAME))
    # completed-unreconciled 行在 REQUEST_COMPLETE 边界立即写出；避免结束
    # 时由调度器聚合全量 completed ledger。
    ledger_sink = (
        _JsonlSink(os.path.join(args.bridge_dir, "ledger.jsonl"))
        if args.sensing else None)
    # 步骤 1-9:真实策略(关感知,默认);决策日志逐批写 online_decision_log.jsonl。
    # 阶段 3:--sensing 开启感知(分层账本 + 两层剩余负载查询;查询/审计
    # 输入,不进策略判据,决策序列与关感知逐字节一致)。
    # 阶段 7 §10.6(joint 仓口径):strategy 模式单变体,无
    # kv_cache_policy 分发——直接构造 Sh30OnlineScheduler(构造器
    # fail-closed 校验;--mode choices 唯一取值 strategy)。
    # joint 仓：三机制联合准入（instance × action 选择 + home/merge
    # 生命周期 + T/E 注入 KV 账本）；开关经 JOINT_* env（§7.1 八组合）。
    scheduler = Sh30OnlineScheduler(
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
    # joint 开关状态落 run 级 provenance（bridge 目录 manifest 侧车；
    # §7.1：manifest 记录每次运行的三机制开关状态及 off 替代模式）。
    # C11 rider（C8-BLOCKED① 收口 + F7 注入事实）：
    # - link_telemetry_injected = SH_LINK_TELEMETRY=1（joint_runner
    #   --quota aimd 的自动注入交接变量；manifest 记录注入事实）；
    # - 遥测完备性五键（epoch/sample 计数、键缺席/窗口破损旗标、
    #   collective_coverage 终值）在 run 结束（verify_run_end 之后）
    #   经 _update_joint_manifest_telemetry 回填——完备性是 run 全程
    #   才能判定的事实（sticky 旗标），侧车读-改-写原子更新。
    joint_manifest_path = os.path.join(
        args.bridge_dir, "joint_mechanism_manifest.json")
    _manifest_payload = scheduler.joint_config.manifest_dict()
    _manifest_payload["link_telemetry_injected"] = (
        os.environ.get("SH_LINK_TELEMETRY", "0") == "1")
    with open(joint_manifest_path, "w", encoding="utf-8") as sink:
        json.dump(_manifest_payload, sink, indent=1, sort_keys=True)
    # Scheduler 已提取构造期所需字段；不要让 main() 的局部变量再额外钉住
    # 完整 manifest。
    del manifest

    # Official peer is FileDecisionBridge: request files are exactly
    # nlohmann::json::dump() bytes (compact, no trailing newline).
    server = BridgeServer(
        args.bridge_dir, canonical_request_producer=True)
    # 终审-中4（kimi 三审）+ 四审-低3/低8：joint 台账侧车改 **try/finally
    # 全路径落盘**（K6 语义"deep_gap 落账 = run 终止"——RED run 经异常
    # 退出也须落终态缺口现场）；落盘函数为模块级（可单测固化）且
    # **原子写**（tmp + os.replace——半截 JSON 属响亮失败而非静默，此处
    # 再消除触发面）。
    _ledgers_path = os.path.join(
        args.bridge_dir, "joint_kv_ledgers.json")
    try:
        result = server.serve_forever(
            scheduler.on_decision_batch,
            on_commit_ack=scheduler.on_commit_ack,
        )
        if result != 0:
            raise RuntimeError("serve_forever returned {}".format(result))

        # 运行结束校验(fail-closed):
        scheduler.verify_run_end()
    finally:
        dump_joint_kv_ledgers(scheduler, _ledgers_path)
    # C11 rider：遥测完备性五键回填 manifest 侧车（verify 之后——
    # sticky 事实 run 全程判定；C8-BLOCKED① 收口）。
    _update_joint_manifest_telemetry(scheduler, joint_manifest_path)
    # 阶段 4 §7.3:每决策批扫描条目数 profile(验收:与总 request 数无关,
    # full_scan_entries 恒为 0)——M3 起已在 build_graph_batch 逐行流式
    # 写出,此处不再结束一次性写出。

    # 阶段 6 §9.1:online_stats.jsonl -- 分项计数器统一采集(Python 侧)。
    # 每已应用交付一行(合并桥接层每 request 服务时间) + 一行汇总。
    # B3(2026-08-28):主变体行已流式写入 .partial(未合并 processing_ns),
    # 此处逐行读回、合并、写终文件后删除暂存——行不再驻留内存,终文件
    # 字节与改前结束一次性写出一致。
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

    # online_decision_log / train_ledger 已由 _JsonlSink 逐行落盘(M3)。
    # joint 台账侧车已改 serve/verify 的 try/finally 全路径落盘（终审-
    # 中4：RED run 异常退出也落终态缺口现场），此处不再重复 dump。
    if args.sensing:
        # 阶段 3:分层账本导出(结束总账核对与差异报告输入;感知关闭的
        # 正式跑不经过这里)。B3:sensing_query_log 已流式落盘,不再结束
        # 一次性写出。
        scheduler.dump_ledger(os.path.join(args.bridge_dir, "ledger.jsonl"))
    # B2/M3/B3:常驻 fd 显式关闭(serve 期之外无写入;异常路径由进程退出
    # 兜底)。digest 关闭态为 None、profile 为 _NullSink,同样安全。
    for sink in (digest_sink, decision_log_sink, train_ledger_sink,
                 profile_sink, sensing_query_sink, ledger_sink):
        if sink is not None:
            sink.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- CLI 顶层 fail-closed
        print("online_service: fatal: {}: {}".format(type(exc).__name__, exc),
              file=sys.stderr)
        sys.exit(1)
