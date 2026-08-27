#!/usr/bin/env python3
"""WP7 负载不均衡度（离线重建，五仓逐字节相同，标准库实现）。

算法严格按主规格 §1.5-A：

    逐请求 (instance, admission→drain 区间)
      → 各 instance 积压时间序列（桶长 = manifest imbalance_bucket_ns）
      → 时间平均 CV 与 Max/Mean；不做账本 dump。

区间语义（逐仓字段映射，全部来自基线产物事实核对）：
  * admission = decision log kind=prefill 行的 tick（S1 另有显式
    admission_time_ns 字段，二者在基线中相等，优先取显式字段）；
  * instance  = decision log kind=decode 行的 decision.decode_instance_index
    （请求实际占用的是其 decode 实例；PD 迁移仓的 prefill 段归属不重复
    计——固定算法只取一个 (instance, interval) 对/请求）；
  * drain     = train_ledger 中 drains 数组含该 request_id 的记录 tick。
    （train_ledger 是 graph batch 账本；本脚本只读取 drains/exits 归属，
    不 dump 账本内容。）

统计口径：每实例时间平均积压 b̄_i = Σ(桶内积压×桶长)/总跨度；
CV = stdev_population(b̄_i)/mean(b̄_i)；Max/Mean = max(b̄_i)/mean(b̄_i)。
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


def collect_intervals(run_dir: Path, repo_variant: str) -> list[dict]:
    """逐请求 (instance, admission→drain) 区间重建。"""
    admissions: dict[str, int] = {}
    decode_instances: dict[str, Optional[int]] = {}
    for record in iter_jsonl(run_dir / DECISION_LOG_RELPATH):
        kind = record.get("kind")
        request_id = record.get("request_id")
        if not request_id:
            continue
        decision = record.get("decision") or {}
        tick = record.get("tick")
        if not isinstance(tick, int):
            fail(f"decision log 行缺整数 tick（kind={kind!r}, "
                 f"request={request_id!r}）")
        if kind == "prefill":
            admission = tick
            for field in EXPLICIT_ADMISSION_FIELDS.get(repo_variant, ()):
                value = decision.get(field)
                if isinstance(value, int):
                    admission = value
                    break
            if request_id in admissions:
                fail(f"请求 {request_id} 出现多次 prefill 决策（准入应恰一次）")
            admissions[request_id] = admission
        elif kind == "decode":
            instance = decision.get("decode_instance_index")
            if not isinstance(instance, int):
                fail(f"请求 {request_id} 的 decode 决策缺整数 "
                     f"decode_instance_index（逐仓字段映射失效？repo="
                     f"{repo_variant}）")
            decode_instances[request_id] = instance

    drains: dict[str, tuple[int, int]] = {}  # request_id -> (tick, instance)
    skipped_first_step_rows = 0
    for record in iter_jsonl(run_dir / TRAIN_LEDGER_RELPATH):
        # WP9 首步批账本行（first_step=true，SPLIT=ON 发射边界记录）不是
        # 终态 drain：真实 drain 在余量批行（同 train_id 的非 first_step
        # 行）。跳过以保持"每请求恰一次 drain"不变量（2026-08-27 B3
        # 集成验证发现：拆分列车的 first_step 行也携带 drains 数组）。
        if record.get("first_step"):
            skipped_first_step_rows += 1
            continue
        drained = record.get("drains")
        if not isinstance(drained, list):
            fail("train_ledger 记录缺 drains 数组（schema 不符）")
        tick = record.get("tick")
        instance = record.get("instance_index")
        if not isinstance(tick, int) or not isinstance(instance, int):
            fail("train_ledger 记录缺整数 tick/instance_index（schema 不符）")
        for request_id in drained:
            if not isinstance(request_id, str):
                fail("train_ledger drains 成员必须是 request_id 字符串")
            if request_id in drains:
                fail(f"请求 {request_id} 被多条列车 drain（应恰一次）")
            drains[request_id] = (tick, instance)

    intervals = []
    skipped_no_admission = []
    skipped_no_decode = []
    skipped_no_drain = []
    skipped_bad_order = []
    instance_mismatch = 0
    for request_id, admission in sorted(admissions.items()):
        instance = decode_instances.get(request_id)
        drain = drains.get(request_id)
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
    for label, skipped in (("无 decode 决策", skipped_no_decode),
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


def instance_timeavg_backlog(intervals: list[dict], bucket_ns: int,
                             span_start: int, span_end: int
                             ) -> dict[int, tuple[float, int]]:
    """各 instance 积压时间序列 → 时间平均积压（返回 instance -> (b̄, n_req)）。

    桶 j 覆盖 [span_start + j*bucket, +bucket)；请求区间对桶的覆盖按
    时间重叠 >0 计入（区间端点闭开 [admission, drain)）。
    """
    n_buckets = max(1, math.ceil((span_end - span_start) / bucket_ns))
    per_instance: dict[int, list[int]] = {}
    counts: dict[int, int] = {}
    for interval in intervals:
        series = per_instance.setdefault(interval["instance"],
                                         [0] * n_buckets)
        counts[interval["instance"]] = counts.get(
            interval["instance"], 0) + 1
        first = (interval["admission_ns"] - span_start) // bucket_ns
        last = (interval["drain_ns"] - span_start) // bucket_ns
        for b in range(max(0, int(first)), min(n_buckets, int(last) + 1)):
            bucket_start = span_start + b * bucket_ns
            bucket_end = bucket_start + bucket_ns
            if interval["drain_ns"] > bucket_start and \
                    interval["admission_ns"] < bucket_end:
                series[b] += 1
    result: dict[int, tuple[float, int]] = {}
    total = span_end - span_start
    if total <= 0:
        fail("load_imbalance：时间跨度为 0，无法做时间平均")
    for instance, series in per_instance.items():
        covered = sum(series) * bucket_ns
        # 末桶可能被 span 截断：按实际桶数×桶长覆盖，误差 ≤1 桶（记录于
        # 输出的 bucket_ns/span，供交叉核对）。
        result[instance] = (covered / total, counts[instance])
    return result


def cmd_load_imbalance(args: argparse.Namespace) -> int:
    manifest = load_slo_manifest(args.manifest or default_manifest_path())
    bucket_ns = require_param_int(manifest, "imbalance_bucket_ns")
    if bucket_ns <= 0:
        fail("imbalance_bucket_ns 必须为正整数（ns）")
    repo_variant = detect_repo_variant(args.run_dir, args.repo_variant)
    intervals = collect_intervals(args.run_dir, repo_variant)
    span_start = min(i["admission_ns"] for i in intervals)
    span_end = max(i["drain_ns"] for i in intervals)
    stats = instance_timeavg_backlog(intervals, bucket_ns, span_start,
                                     span_end)
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
                     "per-instance backlog series -> time-averaged "
                     "CV and Max/Mean (spec 1.5-A)",
        "bucket_ns": bucket_ns,
        "span_start_ns": span_start,
        "span_end_ns": span_end,
        "n_instances": n_instances,
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
