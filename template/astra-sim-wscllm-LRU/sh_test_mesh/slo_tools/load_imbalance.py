#!/usr/bin/env python3
"""WP7 负载不均衡度（离线重建，标准库实现）。

同步基线注记（2026-09-26）：本文件与 face/wscllm 家族同名脚本已按仓分叉
（本仓先行 METRIC-01 精确时间比例积分），不再跨仓逐字节相同。

算法严格按主规格 §1.5-A：

    逐请求 (instance, admission→drain 区间)
      → 各 instance 积压时间积分（时间比例精确积分；imbalance_bucket_ns
        仅作名义桶长登记）
      → 全部配置实例上的时间平均 CV 与 Max/Mean；不做账本 dump。

区间语义（逐仓字段映射，全部来自基线产物事实核对）：
  * admission = decision log kind=prefill 行的 tick（S1 另有显式
    admission_time_ns 字段，二者在基线中相等，优先取显式字段）；
  * instance  = decision log kind=decode 行的 decision.decode_instance_index
    （请求实际占用的是其 decode 实例；PD 迁移仓的 prefill 段归属不重复
    计——固定算法只取一个 (instance, interval) 对/请求）；
  * drain     = train_ledger 中非 first_step 的 batch_train 行 exits 数组
    含该 request_id 的行 tick（D 侧终态行）。
    （train_ledger 是 graph batch 账本；本脚本只读取 exits 归属，不 dump
    账本内容。2026-09-04 口径修正：原读 drains 行——WP9 拆分格式下请求
    在准入 tick 即被退化 prefill_train 行 drain（wsc_llm_online_scheduler
    准入路径同步拼 P 侧行），admission→drain 区间 100% 零长度，wscllm
    全仓 time_avg_backlog 退化为 0；exits 终态行在五仓账本格式下均每请求
    恰一行，0902 批次实测 0 重复/0 缺失/0 零长度区间。）

统计口径（METRIC-01 修正，2026-09-26；2026-09-29 分母来源修正）：每实例
时间平均积压 b̄_i = Σ_r (drain_r − admission_r) / 总跨度——闭开区间
[admission, drain) 重叠时长的精确时间比例积分（∫in-flight dt 恒等式），
不再按「覆盖桶数×桶长」近似（部分覆盖桶按整桶计会高估）；分母=全部配置
实例——配置实例总数的权威来源是 trace_config.csv 的 inference_group 组
行数（本仓标准布局 9 组 × 6 rank；plan_materializer 的 manifest 不携带
实例/组清单，request manifest 的 prefill_ranks 是每实例 rank 数而非实例
数；部署实例连续编号 0..N−1），全程空闲实例 b̄=0 计入总体——只对出现过
的实例求 CV 会从分母移除空闲实例、低估不均衡。bucket_ns 不参与积分，仅
随输出登记（名义桶长）。CV = stdev_population(b̄_i)/mean(b̄_i)；
Max/Mean = max(b̄_i)/mean(b̄_i)。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slo_common import (  # noqa: E402
    DECISION_LOG_RELPATH, TRAIN_LEDGER_RELPATH, default_manifest_path,
    detect_repo_variant, emit_json, fail, iter_jsonl, load_slo_manifest,
    open_output, require_param_int, run_main, write_csv,
)
# METRIC-01：配置实例总数的权威来源 = trace_config.csv 的 inference_group
# 组行数（解析与候选链复用 hbm_watermark，单一口径不另立推导）；每实例
# NPU 数（prefill_ranks 长度）是 hbm_watermark.load_npus_per_instance 的
# 语义，与本工具分母无关。
from hbm_watermark import count_configured_instances  # noqa: E402

# 逐仓 admission 字段分支：S1/S3 的 prefill decision 携带显式
# admission_time_ns；FACE/W 只有 tick（= 准入决策时刻）。基线核对：
# S1 admission_time_ns == tick（1557533114）；FACE estimated_arrival_ns
# == tick（1557524140）。
EXPLICIT_ADMISSION_FIELDS: dict[str, tuple[str, ...]] = {
    "astra-sim-sh_1.0": ("admission_time_ns",),
    "astra-sim-sh_3.0": ("admission_time_ns",),
    "astra-sim-face": (),
    "astra-sim-wscllm": (),
    "astra-sim-sh_2.0": (),
}


def li_prepare() -> dict:
    """decision-log 扫描态（A4：driver 单遍复用；CLI 路径同构）。"""
    return {"admissions": {}, "decode_instances": {}}


def li_consume_decision(record: dict, repo_variant: str, state: dict) -> None:
    """单条决策记录的 admission/decode 归集（循环体逐语句等价）。"""
    admissions = state["admissions"]
    decode_instances = state["decode_instances"]
    kind = record.get("kind")
    request_id = record.get("request_id")
    if not request_id:
        return  # 原循环体的 continue（无 request_id 行不参与归集）
    decision = record.get("decision") or {}
    tick = record.get("tick")
    # bool 是 int 子类：True/False 不得冒充 tick（与下游 instance 域检查
    # 同纪律，显式排除）。
    if isinstance(tick, bool) or not isinstance(tick, int):
        fail(f"decision log 行缺整数 tick（kind={kind!r}, "
             f"request={request_id!r}）")
    if kind == "prefill":
        admission = tick
        for field in EXPLICIT_ADMISSION_FIELDS.get(repo_variant, ()):
            value = decision.get(field)
            if isinstance(value, int) and not isinstance(value, bool):
                admission = value
                break
        if request_id in admissions:
            fail(f"请求 {request_id} 出现多次 prefill 决策（准入应恰一次）")
        admissions[request_id] = admission
    elif kind == "decode":
        instance = decision.get("decode_instance_index")
        if isinstance(instance, bool) or not isinstance(instance, int):
            fail(f"请求 {request_id} 的 decode 决策缺整数 "
                 f"decode_instance_index（逐仓字段映射失效？repo="
                 f"{repo_variant}）")
        if request_id in decode_instances:
            # 同请求多条 decode 决策不再 last-wins（实例归属取决于行序、
            # 掩盖账本重复）——与上方 prefill 判重及 hbm_watermark 的
            # (request_id, kind) 判重同口径。
            fail(f"请求 {request_id} 出现多次 decode 决策（应恰一次）")
        decode_instances[request_id] = instance


def li_collect_drains(run_dir: Path) -> tuple[dict, int]:
    """train_ledger 读（仅本工具使用，保留独立读；读数仍 1 次）。"""
    drains: dict[str, tuple[int, int]] = {}  # request_id -> (tick, instance)
    skipped_first_step_rows = 0
    for record in iter_jsonl(run_dir / TRAIN_LEDGER_RELPATH):
        # WP9 首步批账本行（first_step=true，SPLIT=ON 发射边界记录）不是
        # 终态 drain：真实 drain 在余量批行（同 train_id 的非 first_step
        # 行）。跳过以保持"每请求恰一次 drain"不变量（2026-08-27 B3
        # 集成验证发现：拆分列车的 first_step 行也携带 exits 数组）。
        if record.get("first_step"):
            skipped_first_step_rows += 1
            continue
        train_id = record.get("train_id")
        if not isinstance(train_id, str) or not train_id:
            fail("train_ledger 记录缺字符串 train_id（schema 不符）")
        if train_id.startswith("prefill_train"):
            # WP9 拆分格式的 P 侧退化列车行在准入 tick 发射并携带 drains
            # ——非终态（终态在 D 侧 batch_train 行的 exits）。非拆分格式
            # 无 prefill_train 行，本分支为空操作。
            continue
        exited = record.get("exits")
        if not isinstance(exited, list):
            fail("train_ledger 记录缺 exits 数组（schema 不符）")
        tick = record.get("tick")
        instance = record.get("instance_index")
        if isinstance(tick, bool) or not isinstance(tick, int) \
                or isinstance(instance, bool) or not isinstance(instance, int):
            fail("train_ledger 记录缺整数 tick/instance_index（schema 不符）")
        for request_id in exited:
            if not isinstance(request_id, str):
                fail("train_ledger exits 成员必须是 request_id 字符串")
            if request_id in drains:
                fail(f"请求 {request_id} 被多条列车 drain（应恰一次）")
            drains[request_id] = (tick, instance)
    return drains, skipped_first_step_rows


def li_assemble_intervals(state: dict, drains: dict,
                          skipped_first_step_rows: int) -> list[dict]:
    """admission/decode/drain 三源合并 + 跳过计数 stderr（顺序原样）。"""
    admissions = state["admissions"]
    decode_instances = state["decode_instances"]
    intervals = []
    skipped_no_admission = []
    skipped_no_decode = []
    skipped_no_drain = []
    skipped_bad_order = []
    instance_mismatch = 0
    # 遍历域 = admission ∪ decode 两源并集：只遍历 admissions 会把"无
    # prefill 准入却有 decode 决策"的请求整行静默消失（区间缺一角、
    # 分位/积分分母失真）——显式进跳过清单留痕。
    for request_id in sorted(set(admissions) | set(decode_instances)):
        admission = admissions.get(request_id)
        instance = decode_instances.get(request_id)
        drain = drains.get(request_id)
        if admission is None:
            skipped_no_admission.append(request_id)
            continue
        if instance is None:
            skipped_no_decode.append(request_id)
            continue
        if drain is None:
            skipped_no_drain.append(request_id)
            continue
        drain_tick, drain_instance = drain
        if drain_instance != instance:
            instance_mismatch += 1
        if drain_tick < admission:
            skipped_bad_order.append(request_id)
            continue
        intervals.append({
            "request_id": request_id,
            "instance": instance,
            "admission_ns": admission,
            "drain_ns": drain_tick,
        })
    if skipped_first_step_rows:
        print(f"[load-imbalance] 跳过 WP9 first_step 账本行: "
              f"{skipped_first_step_rows} 行（发射边界记录，非终态 drain）",
              file=sys.stderr)
    for label, skipped in (("无 prefill 准入", skipped_no_admission),
                           ("无 decode 决策", skipped_no_decode),
                           ("无 drain 记录", skipped_no_drain),
                           ("drain 早于 admission", skipped_bad_order)):
        if skipped:
            print(f"[load-imbalance] 跳过（{label}）: {len(skipped)} 例 "
                  f"如 {skipped[:3]}", file=sys.stderr)
    if instance_mismatch:
        print(f"[load-imbalance] decode 实例与 drain 列车实例不一致 "
              f"{instance_mismatch} 例（以 decode_instance_index 为准）",
              file=sys.stderr)
    if not intervals:
        fail("load_imbalance：没有任何可用的 (instance, admission→drain) 区间")
    return intervals


def collect_intervals(run_dir: Path, repo_variant: str) -> list[dict]:
    """逐请求 (instance, admission→drain) 区间重建（公开 API 入口）。

    生产路径（CLI cmd_load_imbalance / driver load_imbalance_finish）已改
    拆分面（li_prepare/li_consume_decision/li_collect_drains/li_assemble_
    intervals 单遍复用），本整读包装不再被生产消费——保留入口供测试与
    外部调用（内部即同一拆分面的串装，语义一致）。"""
    state = li_prepare()
    for record in iter_jsonl(run_dir / DECISION_LOG_RELPATH):
        li_consume_decision(record, repo_variant, state)
    drains, skipped_first_step_rows = li_collect_drains(run_dir)
    return li_assemble_intervals(state, drains, skipped_first_step_rows)


def instance_timeavg_backlog(intervals: list[dict], span_start: int,
                             span_end: int,
                             n_configured: Optional[int] = None
                             ) -> dict[int, tuple[float, int]]:
    """各 instance 积压 → 时间平均积压（返回 instance -> (b̄, n_req)）。

    METRIC-01（2026-09-26）：b̄_i = Σ_r (drain_r − admission_r) / 总跨度
    ——时间比例精确积分，区间 [admission, drain) 闭开重叠时长逐区间可加
    （∫ in-flight dt 恒等式）。取代旧「覆盖桶数×桶长」近似：部分覆盖的
    末桶按整桶计，每区间最多高估一整桶，下游 CV 随区间数叠加。

    n_configured=配置实例数（emit 传 trace_config.csv 的 inference_group
    组行数；None=仅区间内出现的实例）：分母必须含全部配置实例——全程
    空闲实例 b̄=0 计入 CV/Max-Mean 总体，缺席会低估不均衡；区间出现配置
    外实例属 decision log 与 trace_config 不一致，fail-closed。

    2026-08-30 爆内存根治（沿用）：不预分配每实例稠密分桶序列
    （全量 tracelab 曾达 182 GiB），每实例只持一个整数累加器，
    内存 O(区间数)，与时间跨度/桶长彻底解耦。
    """
    total = span_end - span_start
    if total <= 0:
        fail("load_imbalance：时间跨度为 0，无法做时间平均")
    busy_ns: dict[int, int] = {}
    counts: dict[int, int] = {}
    for interval in intervals:
        instance = interval["instance"]
        if n_configured is not None and not 0 <= instance < n_configured:
            fail(f"load_imbalance：区间实例 {instance} 超出配置实例域 "
                 f"[0,{n_configured})——decision log 与 trace_config 配置"
                 f"实例布局不一致（实例须连续编号 0..N-1）")
        busy_ns[instance] = busy_ns.get(instance, 0) + \
            interval["drain_ns"] - interval["admission_ns"]
        counts[instance] = counts.get(instance, 0) + 1
    instances = range(n_configured) if n_configured is not None \
        else sorted(busy_ns)
    return {instance: (busy_ns.get(instance, 0) / total,
                       counts.get(instance, 0))
            for instance in instances}


def load_imbalance_bucket_ns(manifest: dict) -> int:
    """A4 driver 用：manifest 参数读取（保持步骤内时点，红线 R7）。"""
    bucket_ns = require_param_int(manifest, "imbalance_bucket_ns")
    if bucket_ns <= 0:
        fail("imbalance_bucket_ns 必须为正整数（ns）")
    return bucket_ns


def load_imbalance_finish(args: argparse.Namespace, repo_variant: str,
                          bucket_ns: int, state: dict) -> int:
    """A4 driver 用：drains 读 + 区间合并 + 输出（顺序与 CLI 一致）。"""
    drains, skipped_first_step_rows = li_collect_drains(args.run_dir)
    intervals = li_assemble_intervals(state, drains, skipped_first_step_rows)
    return load_imbalance_emit(args, repo_variant, bucket_ns, intervals)


def cmd_load_imbalance(args: argparse.Namespace) -> int:
    # CLI 入口（独立运行行为不变；顺序与拆分前逐语句一致）。A4 driver 经
    # load_imbalance_bucket_ns / li_prepare / li_consume_decision /
    # load_imbalance_finish 组合复用同一逻辑（单遍 decision log）。
    manifest = load_slo_manifest(args.manifest or default_manifest_path())
    bucket_ns = load_imbalance_bucket_ns(manifest)
    repo_variant = detect_repo_variant(args.run_dir, args.repo_variant)
    state = li_prepare()
    for record in iter_jsonl(args.run_dir / DECISION_LOG_RELPATH):
        li_consume_decision(record, repo_variant, state)
    return load_imbalance_finish(args, repo_variant, bucket_ns, state)


def load_imbalance_emit(args: argparse.Namespace, repo_variant: str,
                        bucket_ns: int, intervals: list[dict]) -> int:
    span_start = min(i["admission_ns"] for i in intervals)
    span_end = max(i["drain_ns"] for i in intervals)
    # METRIC-01：分母含全部配置实例（配置实例总数=trace_config.csv 的
    # inference_group 组行数，与 hbm_watermark.count_configured_instances
    # 同一权威来源；本仓部署实例连续编号）。配置实例数缺失即 fail-closed
    # ——CV 总体无定义，不得静默退化为"仅出现过的实例"。分母候选链同样
    # 不回退仓内默认 trace_config（allow_repo_default=False）：改布局
    # 跑出的 run 若不带本地 trace_config 副本，默认配置的组行数会让 CV
    # 分母静默膨胀——fail-closed。
    npus = count_configured_instances(args.run_dir, None,
                                      allow_repo_default=False)
    stats = instance_timeavg_backlog(intervals, span_start, span_end,
                                     n_configured=npus)
    values = [stats[i][0] for i in sorted(stats)]
    n_instances = len(values)
    mean = sum(values) / n_instances
    variance = sum((v - mean) ** 2 for v in values) / n_instances  # 总体方差
    std = math.sqrt(variance)
    cv = std / mean if mean > 0 else None
    max_over_mean = max(values) / mean if mean > 0 else None
    out_rows = [(inst, stats[inst][1], f"{stats[inst][0]:.6f}")
                for inst in sorted(stats)]
    stream, close = open_output(args.output, "slo_load_imbalance.csv",
                                args.run_dir)
    try:
        write_csv(stream,
                  ("instance_index", "n_requests", "time_avg_backlog"),
                  out_rows)
    finally:
        if close:
            stream.close()
    payload = {
        "command": "load_imbalance",
        "repo_variant": repo_variant,
        "algorithm": "per-request (instance, admission->drain) -> "
                     "per-instance backlog time integral (exact, "
                     "time-proportional) -> time-averaged CV and "
                     "Max/Mean over ALL configured instances (spec 1.5-A)",
        "bucket_ns": bucket_ns,
        "span_start_ns": span_start,
        "span_end_ns": span_end,
        "n_instances": n_instances,
        "n_instances_configured": npus,
        "n_requests_used": len(intervals),
        "time_avg_backlog_mean": mean,
        "time_avg_backlog_population_std": std,
        "cv_time_avg": cv,
        "max_over_mean_time_avg": max_over_mean,
        "std_convention": "population (ddof=0)",
    }
    print("", file=sys.stderr)
    emit_json(sys.stderr, payload)
    if args.json:
        jstream, jclose = open_output(args.json, "slo_load_imbalance.json",
                                      args.run_dir)
        try:
            emit_json(jstream, payload)
        finally:
            if jclose:
                jstream.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="load_imbalance.py",
        description="WP7 负载不均衡度离线重建（逐请求 admission→drain 区间 "
                    "→ instance 积压时序 → 时间平均 CV 与 Max/Mean）")
    parser.add_argument("run_dir", type=Path,
                        help="运行目录（含 results/online_decision_log.jsonl "
                             "与 results/train_ledger.jsonl）")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="slo_params_manifest.json（默认：脚本同目录）")
    parser.add_argument("-o", "--output", default="",
                        help="每实例 CSV 输出（'-'=stdout；缺省写 "
                             "run_dir/slo_load_imbalance.csv）")
    parser.add_argument("--json", default="",
                        help="汇总 JSON 输出路径（可选）")
    parser.add_argument("--repo-variant", default=None,
                        help="显式指定 repo_variant（默认读 cpp.log init 行）")
    args = parser.parse_args()
    return int(cmd_load_imbalance(args) or 0)


if __name__ == "__main__":
    sys.exit(run_main(main))
