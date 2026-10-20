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
     bridge_dir(每次 GraphBatch 产出顺带写 digest 行;决策行同批追加;
     digest 缺省关:SH_GRAPH_DIGESTS=0,=1 恒写)。

strategy 模式(步骤 1-9 的 Sh30OnlineScheduler):真实策略(关感知)在
在线骨架中运行,决策日志逐批写 online_decision_log.jsonl(决策由策略实时产出)。
"""

import argparse
import json
import os
import signal
import sys
import time
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
# 墙时分段流(2026-10-05 埋点):每已应用交付一行——桥接三段
# (read_parse/handler/write)+ 调度固定阶段五分段 + scheduler_self_ns;
# 桥接层在 response 原子写后落行(eager=False,有行才建文件),逐行
# flush,SIGTERM 存活。
TIMING_LOG_NAME = "timing.jsonl"
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
    close——不再每行 open/close。行格式统一为
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


class _StatsFanoutSink:
    """online_stats 基础行扇出(2026-10-05 埋点):写 .partial 暂存的
    同时把末行副本留进 stash——桥接层在 response 原子写完后经
    timing_row_source 取走,与桥三分段拼接成 timing.jsonl 行。

    严格顺序协议(一轮至多一个在途 request)下每交付恰一行,单槽
    stash 无歧义;重放/异常路径不入 stash(桥层对应 request 同样
    不落 timing 行)。"""

    def __init__(self, inner, stash: dict):
        self._inner = inner
        self._stash = stash
        self.count = 0

    def __call__(self, row: dict) -> None:
        self._inner(row)
        self._stash["last"] = dict(row)
        self.count += 1

    def close(self) -> None:
        self._inner.close()


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
    直接经访问器取行。消费方=测试与未来实验工具（生产主路径不读
    侧车文件）。

    A12'（2026-09-22，§4.3 补遗）：逐键独立导出——单键失败写
    ``<key>_export_error`` 哨兵字符串（其余键不受连坐；kv_delta_
    journal 的 seq 断链 fail-closed 经哨兵显式落盘，消费端按最弱层
    + 明示注记处理），不再被兜底 except 吞成"四键尽失 + 消费端误判
    零结算"。

    M4（2026-09-26，E/容量验证测量日志）：restore_events（C15 后缀
    逐组恢复 issue/complete 审计——issue/complete 的 at_ns 配对得
    恢复窗口下界）与 adaptive_decisions（C15 adaptive 决策披露侧车
    ——决策 at_ns 支撑两阶段决策验证）随本侧车第五、六键落盘。
    getattr 缺省空序列与 copy_handoff 同款（替身 kv_manager 兼容）。
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
        ("restore_events",
         lambda: [dict(event) for event in
                  getattr(scheduler.kv_manager,
                          "restore_events", ())]),
        ("adaptive_decisions",
         lambda: [dict(row) for row in
                  getattr(scheduler.kv_manager,
                          "adaptive_decisions", ())]),
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
        # W5（A3）：背景世界源终态（ledger = 账本权威投影；
        # absent_legacy_peer = 旧对端缺 work_ledger 字段的显式降级——
        # LoadsWorldSource 注册表计数背景，run 全程 sticky）。
        "work_ledger_world_source": getattr(
            scheduler, "_joint_world_source_status", None),
        "work_ledger_field_absent_seen": getattr(
            scheduler, "work_ledger_absent_seen", None),
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


def _install_sigterm_die_model_flush(scheduler) -> None:
    """C2 收割落盘（2026-10-17 收官后专项）：收割看门狗对 online_service.py
    是 pkill -TERM 直杀——serve 循环、try/finally 台账链、oracle 析构全部
    不走，die_destroy_model（含估值缓存 dirty 落盘）从不执行。此处注册
    SIGTERM handler 只补这一件缺失的事：销毁 die 模型（=缓存落盘，若
    DIE_ESTIMATE_CACHE_PATH 未设则 C 侧照旧跳过），随后恢复默认信号
    自我了断——进程死法、serve 循环、台账 flush 纪律均与改前一致。

    实测校正（runR5_C2h1..C2h4）：标准收割口径（TJE + 固定 requests.csv、
    截断 8.47e9）下 Python 在收割前服务决策批但尚未走到估值路径（oracle
    仍 None）——C2 缓存在该口径本就不 dirty，无文件属正确行为而非缺陷；
    本 handler 是对"Python 在截断前已估值"配置（更重负载/更晚截断）的
    保险。C++ 侧 DieModelAdapter 用核心库 PlanCache（内存态、非 C2 面），
    其持久化属新立项。

    重入安全：Python 信号回调只在主线程字节码边界执行——主线程若正
    阻塞在 die_estimate_batch 的 ctypes 调用里，handler 排到该调用返回
    之后才运行，不存在模型销毁与估值调用竞态。_oracle 是普通属性（惰性
    创建在 _library()），getattr 不触发创建；从未创建过 = 无可落盘，
    handler 直接走默认死法。"""
    def _flush_and_die(signum, frame):  # noqa: ANN001,ARG001
        try:
            # 诊断/运维轨迹（收割轮 python.log 唯一留下的东西）：
            # enter→closed 成对出现 = 落盘路径走通；缺失即定位断点。
            os.write(2, "[C2SIGTERM] enter path={!r} oracle={!r}\n".format(
                os.environ.get("DIE_ESTIMATE_CACHE_PATH"),
                type(getattr(scheduler, "_oracle", None)).__name__).encode())
            oracle = getattr(scheduler, "_oracle", None)
            if oracle is not None:
                oracle.close()
            os.write(2, b"[C2SIGTERM] closed\n")
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGTERM)

    signal.signal(signal.SIGTERM, _flush_and_die)


def _effective_thermal_epoch_ns() -> int:
    """Read the C++ periodic thermal-epoch switch (mirrors ``main_online.cc``).

    Unset / empty = 0 (off).  A non-integer or negative value fails closed,
    exactly as the C++ side does; the returned value is the epoch in simulated
    nanoseconds.  ``SH_THERMAL_EPOCH_NS`` > 0 arms the C++ periodic event plus
    the ``thermal_activity`` provider / ``thermal_cap`` consumer.
    """
    raw = os.environ.get("SH_THERMAL_EPOCH_NS")
    if raw is None or not raw.strip():
        return 0
    text = raw.strip()
    if text.startswith("-"):
        raise SystemExit(
            "SH_THERMAL_EPOCH_NS must be a non-negative integer (got "
            "{!r})".format(raw))
    try:
        value = int(text)
    except ValueError:
        raise SystemExit(
            "SH_THERMAL_EPOCH_NS must be a non-negative integer (got "
            "{!r})".format(raw))
    if value < 0:
        raise SystemExit(
            "SH_THERMAL_EPOCH_NS must be a non-negative integer (got "
            "{!r})".format(raw))
    return value


def _create_thermal_bridge(args, scheduler):
    """Create the run-long thermal session according to the joint thermal switch.

    The session is owned by this service for the whole run (never re-created per
    window, never a CLI subprocess).  Inputs are explicit: the thermal profile,
    the energy table, the policy and the resolver's hardware/mapping exports.

    Fail-closed capability check:

    * ``closed_loop`` needs a live activity source and a live install path.  Both
      are now provided by the C++ periodic thermal epoch (``SH_THERMAL_EPOCH_NS``
      > 0): every epoch the C++ side emits ``thermal_activity`` and consumes the
      ``thermal_cap`` we return.  When that flag is unset / <= 0 there is no
      activity and no consumer, so the run refuses to start instead of reporting
      a loop that could never be closed.
    * ``shadow`` starts and is fully functional as a monitoring arm; when the
      epoch is unset it records explicitly that its activity source is
      unavailable, so the audit cannot be mistaken for real measured activity.

    Returns ``(bridge, status)`` where status is recorded in the run manifest.
    """
    joint_config = getattr(scheduler, "joint_config", None)
    if joint_config is None or not joint_config.thermal_enabled:
        return None, {"thermal_mode": "off"}
    from joint.thermal_bridge import (
        CoarseCapacityInstaller,
        DelegatedCapacityInstaller,
        ThermalBridge,
        load_hardware_and_mapping,
        load_thermal_run_config,
    )
    from joint.thermal_policy import ThermalPolicy, ThermalPolicyConfig

    mode = joint_config.thermal_mode
    config = load_thermal_run_config(Path(joint_config.thermal_run_config))
    if config.mode != mode:
        raise SystemExit(
            "thermal mode mismatch: JOINT_THERMAL_MODE={!r} but the run "
            "configuration declares {!r}".format(mode, config.mode))
    hardware, mapping = load_hardware_and_mapping(config)
    policy = ThermalPolicy(ThermalPolicyConfig.from_json(
        json.loads(Path(config.policy_path).read_text())))
    run_id = os.path.basename(os.path.normpath(args.bridge_dir))
    audit_dir = Path(args.bridge_dir) / "thermal"
    epoch_ns = _effective_thermal_epoch_ns()
    installer = None
    if mode == "closed_loop":
        # Capability check, not a blanket refusal: the install path exists only
        # when the C++ periodic epoch is armed.  Without it the request never
        # carries thermal_activity and the response's thermal_cap is never
        # consumed, so a "closed loop" would be a fiction.
        if epoch_ns <= 0:
            raise SystemExit(
                "thermal_mode=closed_loop requires the C++ periodic thermal "
                "epoch and cap-install path (set SH_THERMAL_EPOCH_NS > 0); "
                "without it there is no thermal_activity source and no "
                "thermal_cap consumer, so a closed_loop run is refused rather "
                "than reporting a loop it cannot close")
        # Activity and install are owned by C++: Python translates each epoch's
        # thermal_activity into the thermal_cap on the response and marks the
        # envelope applied, while the C++ coarse runtime consumes the cap and
        # performs update_capacity.  The delegated installer performs no local
        # install (a cap installed twice would be a bug, not a feature).
        installer = DelegatedCapacityInstaller()
    # With the periodic epoch armed the coarse runtime served counters are the
    # measured activity source; otherwise none is wired yet.
    activity_source = (
        "coarse_runtime_counters_via_thermal_epoch" if epoch_ns > 0
        else "unavailable_until_cpp_epoch")
    bridge = ThermalBridge(
        run_id=run_id, config=config, hardware=hardware, mapping=mapping,
        policy=policy, audit_dir=audit_dir, installer=installer)
    status = {
        "thermal_mode": mode,
        "thermal_run_id": run_id,
        "thermal_hardware_hash": hardware.content_hash,
        "thermal_mapping_hash": mapping.content_hash,
        "thermal_activity_source": activity_source,
        # The effective C++ epoch switch (simulated ns; 0 = no periodic epoch).
        "thermal_epoch_ns": epoch_ns,
        "thermal_audit_dir": str(audit_dir),
    }
    return bridge, status


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
    # 墙时分段流(2026-10-05 埋点):timing.jsonl 每交付一行(桥三段 +
    # 调度五分段);online_stats 基础行经扇出 sink 暂存末行供桥层拼接。
    _timing_stash: dict = {}
    timing_sink = _JsonlSink(
        os.path.join(args.bridge_dir, TIMING_LOG_NAME), eager=False)
    online_stats_sink = _StatsFanoutSink(
        _JsonlSink(
            os.path.join(args.bridge_dir, ONLINE_STATS_PARTIAL_NAME)),
        _timing_stash)
    # completed-unreconciled 行在 REQUEST_COMPLETE 边界立即写出；避免结束
    # 时由调度器聚合全量 completed ledger。
    ledger_sink = (
        _JsonlSink(os.path.join(args.bridge_dir, "ledger.jsonl"))
        if args.sensing else None)
    # W1(2026-09-29)：因果运行账本侧车（每交付一行应用摘要，流式有界；
    # 仿 dump_joint_kv_ledgers 的诊断通道定位——尽力而为，不阻断主路径）。
    work_ledger_sink = _JsonlSink(
        os.path.join(args.bridge_dir, "work_ledger_events.jsonl"),
        eager=False)
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
        # W5（2026-09-29）：构造期显式转发（W1 期间的构造后注入过渡行已
        # 删——见 Sh30OnlineScheduler.__init__ 注释）。
        work_ledger_sink=work_ledger_sink,
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
    # W5（2026-09-29，W3 §7 manifest 接线 + W6 §5 装配点）：预测配置
    # manifest（七正交轴 + params + cross_validate 披露）并入 run 级
    # provenance 的 prediction 节——W3 交付的 joint_prediction_manifest
    # 属性写点（本文件当时属冻结清单，归 W5 接线）。
    _manifest_payload["prediction"] = scheduler.joint_prediction_manifest
    with open(joint_manifest_path, "w", encoding="utf-8") as sink:
        json.dump(_manifest_payload, sink, indent=1, sort_keys=True)
    # Scheduler 已提取构造期所需字段；不要让 main() 的局部变量再额外钉住
    # 完整 manifest。
    del manifest

    # C2 收割落盘（2026-10-17）：serve 循环启动前装好 SIGTERM 优雅化——
    # 收割口径（run_generic.sh 看门狗 pkill -TERM）下补 die 模型销毁
    # （=估值缓存落盘），其余死法/台账纪律零变化（见安装函数注释）。
    _install_sigterm_die_model_flush(scheduler)

    # 热协同仿真会话（thermal_joint_integration_plan）：按 JOINT_THERMAL_MODE
    # 持有单一 run 级热会话；off/未配置时为零行为（不创建、不写审计）。
    # 其状态并入 run manifest，随 serve 生命周期存续、finally 收尾。
    thermal_bridge, _thermal_status = _create_thermal_bridge(args, scheduler)
    # Phase D (Python side): when a thermal session exists, attach the
    # run-long thermal-epoch service to the scheduler so a request carrying
    # ``thermal_activity`` gets its ``thermal_cap`` on the response.  off / no
    # bridge => the service stays None and any activity-bearing request records
    # a cap absence and continues safely.
    if thermal_bridge is not None:
        from joint.thermal_epoch import ThermalEpochService

        scheduler.thermal_epoch_service = ThermalEpochService(thermal_bridge)
    if _thermal_status.get("thermal_mode", "off") != "off":
        _manifest_payload_thermal = None
        try:
            with open(joint_manifest_path, "r", encoding="utf-8") as source:
                _manifest_payload_thermal = json.load(source)
            _manifest_payload_thermal["thermal"] = _thermal_status
            tmp = joint_manifest_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as sink:
                json.dump(_manifest_payload_thermal, sink, indent=1, sort_keys=True)
            os.replace(tmp, joint_manifest_path)
        except Exception:  # noqa: BLE001 -- 诊断通道不得阻断主路径
            traceback.print_exc()

    # Official peer is ShmDecisionBridge: request frame payloads are exactly
    # nlohmann::json::dump() bytes (compact, no trailing newline).
    def _timing_row_source():
        # 扇出 stash 取走即弃(单槽;严格顺序协议下一 request 恰一行)。
        return _timing_stash.pop("last", None)

    def _work_ledger_bytes_source(sequence):
        # BridgeServer writes the exact immutable work_ledger bytes retained
        # by the scheduler's just-accepted response-cache entry (v4 N-line:
        # the raw segment bytes of the accepted delivery). The strict
        # one-in-flight protocol makes this a bounded single-delivery lookup.
        # The journal note (snapshot_kind/event_watermark for the stripped
        # marker) rides along from the apply receipt.
        cached = scheduler._delivery_reply_cache
        if cached is None or cached.get("seq") != sequence:
            return None
        note = getattr(scheduler, "_last_work_ledger_note", None)
        if note is None:
            raise RuntimeError(
                "work_ledger journal note missing for delivery {}".format(
                    sequence))
        return cached.get("work_ledger_canonical"), note

    server = BridgeServer(
        args.bridge_dir, canonical_request_producer=True,
        work_ledger_bytes_source=_work_ledger_bytes_source,
        timing_sink=timing_sink, timing_row_source=_timing_row_source)
    # 终审-中4（kimi 三审）+ 四审-低3/低8：joint 台账侧车改 **try/finally
    # 全路径落盘**（K6 语义"deep_gap 落账 = run 终止"——RED run 经异常
    # 退出也须落终态缺口现场）；落盘函数为模块级（可单测固化）且
    # **原子写**（tmp + os.replace——半截 JSON 属响亮失败而非静默，此处
    # 再消除触发面）。
    _ledgers_path = os.path.join(
        args.bridge_dir, "joint_kv_ledgers.json")
    # 服务墙时锚点(2026-10-05 埋点):serve_forever 全程宿主墙时——
    # 汇总行 service_wall_ns(run 净决策服务时长,不含 plan 加载/收尾)。
    _service_wall_t0 = time.perf_counter_ns()
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
        # 热会话收尾：审计落盘 + 释放（异常/结束路径都必须走；off 时为 None）。
        if thermal_bridge is not None:
            try:
                thermal_bridge.write_audit()
                # Phase D: the per-epoch activity->cap outcome ledger
                # (thermal_epochs.jsonl) lands beside the thermal audit.
                _thermal_service = getattr(
                    scheduler, "thermal_epoch_service", None)
                if _thermal_service is not None:
                    _thermal_service.write_audit(
                        _thermal_status.get("thermal_audit_dir")
                        or (Path(args.bridge_dir) / "thermal"))
            except Exception:  # noqa: BLE001 -- 审计通道尽力而为,不掩主异常
                traceback.print_exc()
            finally:
                thermal_bridge.close()
    _service_wall_ns = time.perf_counter_ns() - _service_wall_t0
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
    # 墙时分段/计数埋点(2026-10-05):投影构建+前推粗相三段与按阶段
    # 投影计数的 run 级聚合——与 joint_decision_metrics 收尾行同源。
    _projection_cost_summary = scheduler._joint_projection_cost_summary()
    # ACC（验收纪律第 3 类，v2.0 §19.3）：预测器自身开销 run 末快照
    # 入 online_stats 汇总行——决策墙时 P50/P95/P99（P/D 分列）、总
    # 占比（整数 ppm）、峰值 RSS、D11 摘要维护费、后端事件量；决策
    # 日志大小在本文件落定后以字节回填（None = 日志未产出，不猜）。
    _decision_log_path = os.path.join(args.bridge_dir, DECISION_LOG_NAME)
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
        # ---- 墙时分段与计数埋点(2026-10-05;干净 EOF 时与
        # timing.jsonl / joint_decision_metrics 行同源同刻) ----
        # 服务墙时锚:serve_forever 全程宿主墙时。
        "service_wall_ns": _service_wall_ns,
        # 桥接三段累计:request 读+解析 / handler 执行 / response
        # dumps+原子写。
        "bridge_segment_ns_total": {
            "read_parse_ns": bridge_stats.get("read_parse_ns", 0),
            "handler_ns": bridge_stats.get("handler_ns", 0),
            "write_ns": bridge_stats.get("write_ns", 0),
        },
        "journal_segment_ns_total": {
            "encode_ns": bridge_stats.get("journal_encode_ns", 0),
            "compress_write_ns": bridge_stats.get(
                "journal_compress_write_ns", 0),
            "write_ns": bridge_stats.get("journal_write_ns", 0),
            "inclusive_total_ns": bridge_stats.get("journal_total_ns", 0),
        },
        # 调度固定阶段五分段累计(每已应用交付之和)。
        "scheduler_segment_ns_total": scheduler.delivery_segment_totals(),
        # P 准入 / D 边界决策墙时按"视图+代价模型构建 / 候选枚举"分段。
        "decision_wall_ns_total": (
            scheduler.decision_wall_timing_totals()),
        # 投影构建 + 前推粗相三段累计 + 背景参考运行计数 + 按阶段
        # 投影计数。
        "projection_timing_ns_total": {
            "build_wall_ns_total": _projection_cost_summary.get(
                "build_wall_ns_total", 0),
            "closure_wall_ns_total": _projection_cost_summary.get(
                "closure_wall_ns_total", 0),
            "candidate_world_wall_ns_total": _projection_cost_summary.get(
                "candidate_world_wall_ns_total", 0),
            "reference_world_wall_ns_total": _projection_cost_summary.get(
                "reference_world_wall_ns_total", 0),
            "reference_world_runs": _projection_cost_summary.get(
                "reference_world_runs", 0),
            "projections_by_stage": dict(
                _projection_cost_summary.get("projections_by_stage")
                or {"P": 0, "D": 0}),
        },
        # die 估值计数/墙时:memo 命中/未命中 + ctypes 批量调用。
        "die_estimate": scheduler._joint_die_estimate_stats(),
        "predictor_cost": scheduler.predictor_cost_snapshot(
            decision_log_bytes=(
                os.path.getsize(_decision_log_path)
                if os.path.exists(_decision_log_path) else None)),
    }
    stats_path = os.path.join(args.bridge_dir, ONLINE_STATS_LOG_NAME)
    partial_path = os.path.join(args.bridge_dir, ONLINE_STATS_PARTIAL_NAME)

    def _base_stat_rows():
        # (2026-09-29 face 移植)原 else 分支(缓冲 scheduler.online_stats_
        # rows)已删:本 CLI 恒传 _JsonlSink(基础行进 .partial 暂存),该
        # 分支仓内不可达(缓冲形态仅供基类测试/夹具直读);缺 sink 即装配
        # 损坏,fail-closed 而非静默空流。
        if getattr(scheduler, "online_stats_sink", None) is None:
            raise RuntimeError(
                "online_stats sink is None; this CLI always streams base "
                "rows to {}".format(ONLINE_STATS_PARTIAL_NAME))
        with open(partial_path, "r", encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    yield json.loads(line)

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
                 profile_sink, sensing_query_sink, ledger_sink,
                 work_ledger_sink, timing_sink):
        if sink is not None:
            sink.close()
    # W1:账本侧车收尾行（run 级汇总；旧对端 zero-delivery 也如实记录）。
    _wl_summary_path = os.path.join(
        args.bridge_dir, "work_ledger_events.jsonl")
    _wl = scheduler.joint_work_ledger
    _wl_summary_row = {
        "summary": True,
        "full_load_count": _wl.full_load_count,
        "delta_load_count": _wl.delta_load_count,
        "applied_event_count": _wl.applied_event_count,
        "replayed_event_count": _wl.replayed_event_count,
        "ledger_version": _wl.version,
        "event_watermark": _wl.event_watermark,
        "field_absent_seen": scheduler.work_ledger_absent_seen,
        "conservation": _wl.conservation_report(),
    }
    try:
        with open(_wl_summary_path, "a", encoding="utf-8") as _wl_out:
            _wl_out.write(json.dumps(_wl_summary_row, sort_keys=True) + "\n")
    except OSError:
        traceback.print_exc()  # 诊断通道尽力而为
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- CLI 顶层 fail-closed
        print("online_service: fatal: {}: {}".format(type(exc).__name__, exc),
              file=sys.stderr)
        sys.exit(1)
