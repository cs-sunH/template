#!/usr/bin/env python3
"""online_service.py -- 在线决策服务入口(方案 §4 步骤 1-8 操作 5,face 版)。

CLI:
    python3 online/online_service.py --bridge-dir <dir> --mode strategy \
        --plan-dir <plan/manifest 目录>

流程:
  1. 载入 plan-dir/manifest.json(每 request 全部事实,replay 权威之一)与
     trace_config(load_face_trace_config;--config 可覆盖);
  2. BridgeServer.serve_forever(scheduler.on_decision_batch,
     on_commit_ack=scheduler.on_commit_ack)——阻塞读 req_notify.fifo,
     零 polling;决策异常 => error response + exit 1(fail-closed);
     EOF(C++ 关闭写端)= 运行结束:verify_run_end + 日志消费完整性校验,
     并把 graph_batch_digests.jsonl / online_decision_log.jsonl 写入
     bridge_dir(决策行同批追加;digest 行按 SH_GRAPH_DIGESTS 开关——
     # (2026-09-30) py-online-C3 如实化:生产默认 SH_GRAPH_DIGESTS=0
     不产出 digest 文件,仅显式 =1 时每次 GraphBatch 产出顺带写行,
     旧"恒写"描述与 C2/D2 开关口径不符)。

strategy 模式(步骤 1-9 的 FaceOnlineScheduler):真实策略(关感知)在
在线骨架中运行(决策由策略实时产出)。
"""

import argparse
import json
import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# import 路径:本文件位于 workload/llama2_7b_inference/online/,共享配置与
# 发射模块在上一级。路径只做 import 用途(红线:generate_face_trace.py /
# face_scheduler.py 只读 import 与注释)。
# --------------------------------------------------------------------------
_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _path in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from generate_face_trace import load_face_trace_config  # noqa: E402
from online.decision_bridge import BridgeServer  # noqa: E402
from online.die_model_oracle import (  # noqa: E402
    MANIFEST_COMPUTE_COST_MODEL_KEY,
    DieModelOracle,
    resolve_die_model_config_path,
)
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402


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


def _workspace_admission_mode(config) -> str:
    """Resolve auto/required/off against the selected Die profile.

    coarse_die_v2 carries temporary workspace and must use the v2 capacity
    handshake. Legacy and non-Die runs keep the frozen v1 bridge by default.
    """
    requested = os.environ.get("SH_WORKSPACE_ADMISSION", "auto")
    if requested not in {"auto", "required", "off"}:
        raise ValueError(
            "SH_WORKSPACE_ADMISSION must be auto|required|off, got {!r}"
            .format(requested))
    profile = None
    system_config = getattr(config, "system_config", None)
    if system_config is not None:
        with open(system_config, "r", encoding="utf-8") as source:
            system = json.load(source)
        profile = system.get("die-model-profile")
    coarse_workspace = profile == "coarse_die_v2"
    if requested == "auto":
        return "required" if coarse_workspace else "off"
    if coarse_workspace and requested == "off":
        raise ValueError(
            "coarse_die_v2 requires workspace admission; "
            "SH_WORKSPACE_ADMISSION=off is unsupported")
    if requested == "required" and profile in {
            "mesh_die_v1", "tx8_reference_v1"}:
        raise ValueError(
            "workspace admission is unsupported for legacy Die profile {!r}"
            .format(profile))
    return requested


def _graph_digests_enabled() -> bool:
    """C2/D2(2026-08-28):graph_batch_digests.jsonl 开关,env SH_GRAPH_DIGESTS
    (默认 "0" 关)。digest 行是纯审计产物且每批全量二次序列化
    (nodes+edges 再 dumps+sha256),生产默认关;显式 =1 恢复改前恒写行为。
    (响应字节复用方案被否决:digest 需 sort_keys 口径而桥响应是另一序列
    化格式,复用会改桥协议字节——红线禁改。)"""
    return os.environ.get("SH_GRAPH_DIGESTS", "0") == "1"


class _JsonlSink:
    """jsonl 逐行落盘器(M3 流式落盘 + B2 常驻 fd,2026-08-23)。

    常驻 fd:构造(或首行)时打开一次,__call__ 只 write+flush,结束
    close——不再每行 open/close。行格式 json.dumps(row, sort_keys=True)
    + "\\n",行序不变 ⇒ 与改前结束一次性写出的文件逐字节相同。

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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="face online decision service")
    parser.add_argument("--bridge-dir", required=True,
                        help="bridge FIFO/request-response 目录(与 C++ 共享)")
    parser.add_argument("--mode", default="strategy", choices=("strategy",),
                        help="调度模式:strategy(步骤 1-9;唯一保留模式)")
    parser.add_argument("--plan-dir", required=True,
                        help="plan/manifest 目录(含 manifest.json)")
    parser.add_argument("--config", default=None,
                        help="trace_config.csv 路径(缺省用 workload 默认)")
    parser.add_argument("--sensing", action="store_true", default=False,
                        help="阶段 3 感知开关(默认关,阶段 6 前默认关):分层"
                             "账本最小子集 + 两层剩余负载查询;感知数据是"
                             "查询/审计输入,不进策略判据,决策序列不变")
    parser.add_argument(
        "--die-model-lib", default=None,
        help="阶段 D(face-LRU 随迁):libdie_model_cabi.so 路径(缺省 env "
             "DIE_MODEL_CABI_PATH → 仓内 build 发现序,见 "
             "online/die_model_oracle.py;fail-closed,不回退 roofline)")
    parser.add_argument(
        "--die-model-config", default=None,
        help="阶段 D(face-LRU 随迁):resolved_die_model.json 路径(缺省 env "
             "DIE_MODEL_CONFIG_PATH → 仓内发现 → manifest 的 "
             "die-model-config 键)")
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

    # 阶段 D(2026-10-03,契约 §3.2/§7 裁定 2;face-LRU 随迁):计算成本
    # 后端选择。backend=die(manifest compute-cost-model == "die",或显式
    # --die-model-lib/--die-model-config 选通)时构造 DieModelOracle 注入
    # 调度器——select_decode_instance 的本地计算成本换 die 孤立估值;
    # backend 非 die 完全不走此路(缺 manifest 键 = 旧 roofline 兼容态,
    # 回归保护)。die 后端下库/配置缺位一律 fail-closed,绝不静默回退。
    compute_cost_model = manifest.get(MANIFEST_COMPUTE_COST_MODEL_KEY)
    if compute_cost_model is not None and compute_cost_model not in (
            "die", "roofline", "replay"):
        raise ValueError(
            "manifest {!r} must be one of die|roofline|replay, got {!r}"
            .format(MANIFEST_COMPUTE_COST_MODEL_KEY, compute_cost_model))
    if compute_cost_model in ("roofline", "replay") and (
            args.die_model_lib is not None
            or args.die_model_config is not None):
        raise ValueError(
            "manifest compute-cost-model={!r} conflicts with explicit "
            "--die-model-lib/--die-model-config (die backend)".format(
                compute_cost_model))
    die_backend = (
        compute_cost_model == "die"
        or args.die_model_lib is not None
        or args.die_model_config is not None
    )
    cost_oracle = None
    if die_backend:
        die_config_path = resolve_die_model_config_path(
            explicit=args.die_model_config,
            manifest=manifest,
            plan_dir=args.plan_dir,
        )
        if die_config_path is None:
            raise RuntimeError(
                "die backend requires resolved_die_model.json: pass "
                "--die-model-config, set DIE_MODEL_CONFIG_PATH, materialize "
                "a discoverable copy, or set the manifest die-model-config "
                "key (fail-closed; no roofline fallback)")
        cost_oracle = DieModelOracle(
            lib_path=args.die_model_lib,
            config_json_path=str(die_config_path),
        )
    workspace_admission = _workspace_admission_mode(config)
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
    # 步骤 1-9:真实策略(关感知,默认);决策日志逐批写 online_decision_log.jsonl。
    # 阶段 3:--sensing 开启感知(分层账本 + 两层剩余负载查询;查询/审计
    # 输入,不进策略判据,决策序列与关感知逐字节一致)。
    # face 调度器唯一(B3 2026-09-06:KV 冷热管理内核为唯一行为,契约 §8
    # "行为上新管理器唯一,无档位分支")——2026-09-25 session 级 Tiered-LRU
    # 命名卫生起唯一合法值 session_lru_tiered 分发到 FaceOnlineScheduler
    # (构造器内 fail-closed 校验;分发不依赖任何代码默认值,runner 显式传
    # kv_cache_policy)。
    ledger_sink = None
    if config.kv_cache_policy == "session_lru_tiered":
        from online.face_online_scheduler import (  # noqa: E402
            FaceOnlineScheduler, read_train_max_iter,
        )
        train_max_iter = read_train_max_iter()
        certificate = None
        if workspace_admission == "required":
            from online.manifest_workspace_envelope import (
                derive_manifest_workspace_envelope,
            )
            if cost_oracle is None or not cost_oracle.coarse:
                raise ValueError("workspace certificate requires coarse Die oracle")
            certificate = derive_manifest_workspace_envelope(
                manifest, config, train_max_iter=train_max_iter,
                library_path=cost_oracle.lib_path,
            )
            certificate_path = Path(args.bridge_dir) / "workspace_envelope_certificate.json"
            certificate_path.write_text(certificate.canonical_json + "\n", encoding="utf-8")
            certificate = certificate.without_proof()
        graph = GraphBatchBuilder(
            config, source_release_capacity_gates=workspace_admission == "required")
        # completed-unreconciled 行在 REQUEST_COMPLETE 边界立即写出；避免
        # 结束时由调度器聚合全量 completed ledger。
        ledger_sink = (
            _JsonlSink(os.path.join(args.bridge_dir, "ledger.jsonl"))
            if args.sensing else None)
        scheduler = FaceOnlineScheduler(
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
            cost_oracle=cost_oracle,
            workspace_admission=workspace_admission,
            train_max_iter=train_max_iter,
            prospective_workspace_certificate=certificate,
        )
    else:
        raise ValueError(
            "strategy 模式不支持 kv_cache_policy {!r}".format(
                config.kv_cache_policy))
    # Scheduler 已提取构造期所需字段；不要让 main() 的局部变量再额外钉住
    # 完整 manifest。
    del manifest, certificate

    # Official peer is FileDecisionBridge: request files are exactly
    # nlohmann::json::dump() bytes (compact, no trailing newline).
    server = BridgeServer(
        args.bridge_dir, canonical_request_producer=True,
        workspace_admission=workspace_admission)
    server.serve_forever(
        scheduler.on_decision_batch,
        on_commit_ack=scheduler.on_commit_ack,
    )
    # (2026-10-01 死分支清理)serve_forever 仅在 EOF 时返回 0,错误路径
    # 均经 _fail/sys.exit 非零退出——"非零返回值"分支不可达,已删。

    # 运行结束校验(fail-closed):
    scheduler.verify_run_end()
    # 阶段 4 §7.3:每决策批扫描条目数 profile(验收:与总 request 数无关,
    # full_scan_entries 恒为 0)——M3 起已在 build_graph_batch 逐行流式
    # 写出;profile_sink 恒非 None(_JsonlSink 或 SH_PROFILE_JSONL=0 时的
    # _NullSink),不存在结束一次性写出的回退路径。

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
        # D4 (2026-09-05): 被动驱逐深缺口观测。
        "kv_deep_gap_events": int(getattr(
            getattr(scheduler, "kv_manager", None), "deep_gap_events", 0)),
        # 方案一(2026-10-20)风险一/风险三观测(verify_run_end 结算后
        # 读取;getattr 防御式——无该簿记的变体/夹具保持缺键语义一致,
        # 这里统一落 None 而非炸行):
        #   admission_blocked_streak_histogram 请求级最大连续准入阻塞
        #       批数直方图({streak: 请求数});空 run 为 {}(零阻塞)。
        #   workspace_floor_reconciliation 活跃 workspace 峰值 vs
        #       prospective certificate 地板的对账行(peak/floor/slack
        #       三键逐 rank);非 required 模式为 None(无证书无对账)。
        "admission_blocked_streak_histogram": getattr(
            scheduler, "admission_blocked_streak_histogram", None),
        "workspace_floor_reconciliation": getattr(
            scheduler, "workspace_floor_reconciliation", None),
    }
    if cost_oracle is not None and getattr(cost_oracle, "coarse", False):
        # Design section 12: preserve the original FACE selector composition
        # while identifying it separately from a complete known-world replay.
        summary_row["die_predictor_disclosure"] = {
            "composition_profile": "face_legacy_selector",
            "approximate_predictor": True,
            "local_plan_profile": "coarse_die_v2",
            "representative_tp_rank": 0,
            "collectives_included": False,
            "explicit_queue_delay_included": False,
            "d2d_cost_profile": "fixed_latency_addend",
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
    # P3(face-LRU 审计,2026-09-29):原结束期 `_write_jsonl` 兜底两分支
    # 删除——main 无条件构造 decision_log_sink / train_ledger_sink
    # (_JsonlSink),两守卫恒 False,函数全仓零活调用。
    if args.sensing:
        # 阶段 3:分层账本导出(结束总账核对与差异报告输入;感知关闭的
        # 正式跑不经过这里)。B3:sensing_query_log 已流式落盘,不再结束
        # 一次性写出。
        scheduler.dump_ledger(os.path.join(args.bridge_dir, "ledger.jsonl"))
    # B2/M3/B3:常驻 fd 显式关闭(serve 期之外无写入;异常路径由进程退出
    # 兜底)。digest/profile 关闭态为 None/_NullSink,同样安全。
    for sink in (digest_sink, decision_log_sink, train_ledger_sink,
                 profile_sink, sensing_query_sink, ledger_sink):
        if sink is not None:
            sink.close()
    # 阶段 D(face-LRU 随迁):释放 die 成本库模型句柄(库内 LRU 计划缓存
    # 随句柄销毁)。
    if cost_oracle is not None:
        cost_oracle.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- CLI 顶层 fail-closed
        print("online_service: fatal: {}: {}".format(type(exc).__name__, exc),
              file=sys.stderr)
        sys.exit(1)
