#!/usr/bin/env python3
"""Cross-run metrics post-processing (implementation doc sec.11).

Input: one or more run logs.  Only lines starting with ``[METRIC] `` are
parsed, each holding one single-line JSON record (sec.11.1).  Records are
grouped into runs (one run per ``init`` record occurrence per log).  When a
run's init record carries ``manifest_path``, the referenced
``metrics_manifest.json`` (microbenchmark point spec) and its sibling
``manifest.json`` (model/hardware/kv_policy labels) and
``planner_roofline_stats.json`` (doc sec.8.8 aggregates) are read as the run's
own declared provenance; all metrics themselves come from the log.

Output (sec.11.2/11.3):

- ``raw_metrics.csv``: one row per measured entity with an explicit ``source``
  column — ``simulator`` (service run latency/tput summary),
  ``simulator_microbenchmark`` (per benchmark-point iteration rows),
  ``planner_roofline`` (streaming planner iteration aggregates, sec.8.8), and
  ``planner_memory_ledger`` (per-rank capacity time-average and peak rows,
  sec.7.8/7.7).
- ``normalized_metrics.csv``: per comparison group, the two frozen methods
  (sec.3.6) with full denominator provenance (``normalization_method``,
  ``normalization_group_id``, ``baseline_run_id``, ``denominator_*``).
- ``request_metrics.csv`` (B1/WP1, full detail only): one row per request
  record, joined fail-closed against the run's ``manifest.json`` sidecar
  (sibling of the init record's ``manifest_path``).  Column set/order is
  frozen by the SLO execution plan §2; fields not yet populated by later
  work packages are written as ``NA``.  ``summary``/``off`` runs produce no
  file (a note is printed, which lands in the runner's postprocess.log).
  ``first_token_ns``: exact when the WP9 code-8 events exist; otherwise the
  spec 1.6 train-interpolated proxy (B4 fallback after the B3_S3 60s
  decision-equivalence gate failure, see the ``_TrainProxyIndex`` block)
  tagged ``first_token_source=train_interpolated`` -- display metric only,
  never a SLO judgment input.

Failure conditions (sec.11.4, all abort with a non-zero exit):

1. two conflicting summary records for the same run;
2. explicitly configured comparison group whose workload digests
   (trace_digest / request_mapping_digest) differ without
   ``"allow_digest_mismatch": true``;
3. ``ratio_to_baseline`` selected without exactly one baseline run;
4. a zero normalization denominator;
5. a summary whose completed/incomplete counts are not backed by the run's
   request records (an incomplete request silently counted as complete);
6. mixed metric definition (schema) versions inside one run or one
   comparison group.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from metrics_schema import (  # noqa: E402
    NORMALIZATION_GROUP_MAX,
    NORMALIZATION_RATIO_TO_BASELINE,
    MetricsSchemaError,
    build_normalization_record,
)

METRIC_PREFIX = "[METRIC] "
SUPPORTED_SCHEMA = 1

SOURCE_SIMULATOR = "simulator"
SOURCE_MICROBENCHMARK = "simulator_microbenchmark"
SOURCE_PLANNER_ROOFLINE = "planner_roofline"
SOURCE_PLANNER_MEMORY = "planner_memory_ledger"

RAW_COLUMNS = [
    "run_id",
    "repo_variant",
    "model",
    "dataset",
    "hardware",
    "trace_digest",
    "request_mapping_digest",
    "kv_policy",
    "source",
    "run_mode",
    "schema",
    "completed_requests",
    "incomplete_requests",
    # M15(face-LRU 审计,2026-09-29)披露列:mean/p50/p95/p99_e2e_ns 口径
    # 改为"全部 completed 请求"(ordering 违规者是真实完成,不再静默剔除);
    # 旧 ordering-ok 口径经下列披露列保留可追溯。kv_bin/batch/count/
    # sum/min/max_iteration_time_ns 六列在本仓保留:planner_roofline 记录
    # 的历史写端（metrics_integration.py）已随 2026-10-01 死代码审计删
    # 除（SH3 离线生成入口自 2026-08-18 起拒绝桩、全仓零 import——写端
    # 不存在，"仍生产"为失实表述）；六列与 _planner_roofline_rows 读端
    # 保留消费手工注入/历史日志的兼容面。
    "completed_ordering_violations",
    "mean_e2e_ns",
    "p50_e2e_ns",
    "p95_e2e_ns",
    "p99_e2e_ns",
    "completed_e2e_ordering_ok_mean_ns",
    "completed_e2e_ordering_ok_p99_ns",
    "drain_tput_rps",
    "sim_window_tput_rps",
    "benchmark_point_id",
    "phase",
    "tp_degree",
    "prefill_chunk",
    "decode_batch",
    "kv_length",
    "kv_bin",
    "repeat_index",
    "batch",
    "count",
    "sum_iteration_time_ns",
    "min_iteration_time_ns",
    "max_iteration_time_ns",
    "iteration_start_ns",
    "iteration_end_ns",
    "iteration_time_ns",
    "total_num_ops",
    "total_local_bytes",
    "compute_util",
    "hbm_bw_util",
    "rank",
    "capacity_bytes",
    "resident_capacity_timeavg_util",
    "committed_capacity_timeavg_util",
    "peak_physical_used_bytes",
    "peak_committed_used_bytes",
    "memory_actions_total",
    "memory_actions_replayed",
    "memory_actions_unresolved",
]

NORMALIZED_COLUMNS = [
    "run_id",
    "benchmark_point_id",
    "repo_variant",
    "source",
    "normalization_group_id",
    "normalization_method",
    "baseline_run_id",
    "denominator_time",
    "denominator_tput",
    "denominator_iteration_time",
    "normalized_time",
    "normalized_tput",
    "normalized_iteration_time",
]

# SLO execution plan §2: fixed column set/order for the per-request CSV.
# Later work packages fill values only; no column is ever added.
REQUEST_METRICS_COLUMNS = [
    "queue_index",
    "request_id",
    "session_id",
    "turn_index",
    "request_type",
    "terminal_status",
    "arrival_ns",
    "prefill_start_ns",
    "prefill_end_ns",
    "decode_start_ns",
    "first_token_ns",
    "first_token_source",
    "completion_ns",
    "queue_ns",
    "prefill_ns",
    "prefill_decode_gap_ns",
    "decode_ns",
    "e2e_ns",
    "kv_hit_state",
    "restore_start_ns",
    "restore_complete_ns",
    "pre_prefill_restore_ns",
    "hidden_restore_ns",
    "exposed_restore_stall_ns",
    "hidden_ratio",
    "prefill_length",
    "decode_length",
    "prefix_len",
    "instructions",
]

REQUEST_TYPE_KNOWN = ("human", "tool")


class PostprocessError(RuntimeError):
    """A sec.11.4 failure condition."""


@dataclass
class Run:
    run_id: str
    log_path: Path
    init: dict[str, Any]
    summaries: list[dict[str, Any]] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)
    rank_compute: list[dict[str, Any]] = field(default_factory=list)
    iterations: list[dict[str, Any]] = field(default_factory=list)
    capacity_timeavg: list[dict[str, Any]] = field(default_factory=list)
    planner_peaks: list[dict[str, Any]] = field(default_factory=list)
    planner_roofline: list[dict[str, Any]] = field(default_factory=list)
    consistency: list[dict[str, Any]] = field(default_factory=list)
    schema_versions: set[int] = field(default_factory=set)

    @property
    def summary(self) -> dict[str, Any]:
        if not self.summaries:
            raise PostprocessError(f"run {self.run_id}: no summary record found")
        first = self.summaries[0]
        for other in self.summaries[1:]:
            if other != first:
                raise PostprocessError(
                    f"run {self.run_id}: conflicting summary records "
                    f"(sec.11.4): {json.dumps(first, sort_keys=True)[:200]} vs "
                    f"{json.dumps(other, sort_keys=True)[:200]}"
                )
        return first


def _resolve_log_path(log_path: Path) -> Path:
    """R5-F1（2026-09-29 face 移植）：归档 gz 形态透明回退——README ⑤ 步
    的字面 cpp.log 路径在归档 run_dir 会 "log not found" 必失败。字面
    路径缺失而 <path>.gz 在场时按 gz 解析；两者皆缺时原样返回（由调用
    方报缺）。对齐同链范式（slo_common / ledger_reconcile 的透明 .gz
    回退）。"""
    if log_path.is_file() or str(log_path).endswith(".gz"):
        return log_path
    gz_path = log_path.with_name(log_path.name + ".gz")
    if gz_path.is_file():
        return gz_path
    return log_path


def _parse_logs(log_paths: Sequence[Path]) -> list[Run]:
    runs: list[Run] = []
    for log_path in log_paths:
        log_path = _resolve_log_path(log_path)
        current: Optional[Run] = None
        run_index = 0
        # R5-F1: 归档 gz 形态透明解析（_resolve_log_path 已把字面路径
        # 解析到 <path>.gz；此处按后缀选 opener）。
        opener = gzip.open if str(log_path).endswith(".gz") else open
        with opener(log_path, "rt", encoding="utf-8",
                    errors="replace") as source:
            for line in source:
                if not line.startswith(METRIC_PREFIX):
                    continue
                payload = line[len(METRIC_PREFIX):].strip()
                if not payload:
                    continue
                try:
                    record = json.loads(payload)
                except json.JSONDecodeError as error:
                    raise PostprocessError(
                        f"{log_path}: malformed [METRIC] JSON line: {error}"
                    ) from None
                schema = record.get("schema")
                record_type = record.get("type")
                if record_type == "init":
                    run_index += 1
                    explicit = record.get("run_id") or ""
                    run_id = explicit or f"{log_path.name}#run{run_index}"
                    current = Run(run_id=run_id, log_path=log_path, init=record)
                    runs.append(current)
                elif current is None:
                    raise PostprocessError(
                        f"{log_path}: [METRIC] {record_type!r} record before any "
                        "init record"
                    )
                if schema is not None:
                    current.schema_versions.add(schema)
                if record_type == "init":
                    continue
                if record_type == "summary":
                    current.summaries.append(record)
                elif record_type == "request":
                    current.requests.append(record)
                elif record_type == "rank_compute":
                    current.rank_compute.append(record)
                elif record_type == "iteration":
                    current.iterations.append(record)
                elif record_type == "capacity_timeavg":
                    current.capacity_timeavg.append(record)
                elif record_type == "planner_memory_peaks":
                    current.planner_peaks.append(record)
                elif record_type == "planner_roofline_iteration_stats":
                    current.planner_roofline.append(record)
                elif record_type == "consistency":
                    current.consistency.append(record)
                # memory_anchor and unknown types are tolerated but unused.
    # (2026-10-01 审查修复) 同 run_id 且 summary 全等的重复输入去重：
    # 旧实现只查冲突不去重——同一 log 传两遍时 raw/normalized/request
    # 行翻倍、type_counts 双计（summary 冲突仍 fail-closed）。
    seen: dict[str, Run] = {}
    for run in runs:
        previous = seen.get(run.run_id)
        if previous is not None:
            if previous.summary != run.summary:
                raise PostprocessError(
                    f"run_id {run.run_id!r} appears in multiple logs with "
                    "conflicting summaries (sec.11.4)"
                )
            continue
        seen[run.run_id] = run
    for run in runs:
        if run.schema_versions != {SUPPORTED_SCHEMA}:
            raise PostprocessError(
                f"run {run.run_id}: unsupported or mixed metric schema "
                f"versions {sorted(run.schema_versions)} (sec.11.4); this "
                f"postprocessor supports schema {SUPPORTED_SCHEMA} only"
            )
    return list(seen.values())


def _load_manifest_sidecars(run: Run) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Return (metrics_manifest, service manifest.json, planner_roofline records)."""

    metrics_manifest: dict[str, Any] = {}
    service_manifest: dict[str, Any] = {}
    roofline_records: list[dict[str, Any]] = []
    manifest_path = run.init.get("manifest_path")
    if not manifest_path:
        return metrics_manifest, service_manifest, roofline_records
    path = Path(manifest_path)
    try:
        metrics_manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        metrics_manifest = {}
    try:
        service_manifest = json.loads(
            (path.parent / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        service_manifest = {}
    if not run.planner_roofline:
        try:
            sidecar = json.loads(
                (path.parent / "planner_roofline_stats.json").read_text(encoding="utf-8")
            )
            roofline_records = [
                record
                for record in sidecar.get("records", [])
                if record.get("type") == "planner_roofline_iteration_stats"
            ]
        except (OSError, json.JSONDecodeError):
            roofline_records = []
    return metrics_manifest, service_manifest, roofline_records


def _check_request_completion(run: Run) -> None:
    """sec.11.4: an incomplete request must never be silently counted."""

    summary = run.summary
    if not run.requests:
        return  # summary detail level: nothing to cross-check against
    # R9-F2（2026-09-29 face 移植）：显式布尔判定——bool("false") 恒真
    # 会把字符串化的 failed 计成 completed（与 _request_metric_rows 的
    # `is True` 同口径）。
    completed_records = sum(
        1 for record in run.requests if record.get("completed") is True)
    incomplete_records = len(run.requests) - completed_records
    claimed_completed = summary.get("completed_unique_requests")
    claimed_incomplete = summary.get("incomplete_requests")
    if claimed_completed != completed_records:
        raise PostprocessError(
            f"run {run.run_id}: summary claims completed_unique_requests="
            f"{claimed_completed} but the request records back "
            f"{completed_records} (sec.11.4)"
        )
    if claimed_incomplete != incomplete_records:
        raise PostprocessError(
            f"run {run.run_id}: summary claims incomplete_requests="
            f"{claimed_incomplete} but the request records show "
            f"{incomplete_records} (sec.11.4)"
        )


def _labels(run: Run, service_manifest: dict[str, Any], run_config: dict[str, Any]) -> dict[str, str]:
    # F2_PY fix (2026-09-29, report 3.6 sister-repo sync from astra-sim-wscllm):
    # a well-formed-JSON but malformed-shape manifest.json (e.g. "model":
    # "llama2" -- a string, or "model": null -- an explicit JSON null) used to
    # escape the chained .get calls below as a bare AttributeError, outside
    # the PostprocessError channel that main() is the only catcher of.  The
    # label blocks are comparison-group keys, so a wrong shape aborts through
    # PostprocessError instead of crashing with a traceback.
    for name in ("model", "hardware", "kv_management"):
        block = service_manifest.get(name, {})
        if not isinstance(block, dict):
            raise PostprocessError(
                f"run {run.run_id}: service manifest block {name!r} must "
                f"be an object (got {type(block).__name__} "
                f"{block!r}; fail-closed)"
            )
    model = run_config.get("model") or service_manifest.get("model", {}).get("name", "")
    hardware_block = service_manifest.get("hardware", {})
    hardware = (
        run_config.get("hardware")
        or hardware_block.get("label")
        or hardware_block.get("case")
        or hardware_block.get("capacity_profile")
        or ""
    )
    kv_policy = run_config.get("kv_policy") or service_manifest.get("kv_management", {}).get("policy", "")
    dataset = run_config.get("dataset") or ""
    if not dataset:
        queue_csv = service_manifest.get("request_queue_csv", "")
        if not isinstance(queue_csv, str):
            # RV7-1 (2026-09-29, report 3.6 sister-repo sync from
            # astra-sim-wscllm): same family as the three blocks above -- a
            # non-string (or explicit-null) queue path used to escape as a
            # bare TypeError from Path(); route it through the same
            # PostprocessError channel instead.
            raise PostprocessError(
                f"run {run.run_id}: service manifest request_queue_csv must "
                f"be a string (got {type(queue_csv).__name__} "
                f"{queue_csv!r}; fail-closed)"
            )
        dataset = Path(queue_csv).name if queue_csv else ""
    return {
        "model": str(model),
        "dataset": str(dataset),
        "hardware": str(hardware),
        "kv_policy": str(kv_policy),
    }


def _base_row(run: Run, labels: dict[str, str], source: str) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "repo_variant": run.init.get("repo_variant", ""),
        "model": labels["model"],
        "dataset": labels["dataset"],
        "hardware": labels["hardware"],
        "trace_digest": run.init.get("trace_digest", ""),
        "request_mapping_digest": run.init.get("request_mapping_digest", ""),
        "kv_policy": labels["kv_policy"],
        "source": source,
        "run_mode": run.init.get("run_mode", ""),
        "schema": SUPPORTED_SCHEMA,
    }


def _service_rows(run: Run, labels: dict[str, str]) -> list[dict[str, Any]]:
    summary = run.summary
    row = _base_row(run, labels, SOURCE_SIMULATOR)
    row.update(
        {
            "completed_requests": summary.get("completed_unique_requests", ""),
            "incomplete_requests": summary.get("incomplete_requests", ""),
            # M15(face-LRU 审计,2026-09-29):mean/p50/p95/p99_e2e_ns 口径 =
            # 全部 completed 请求(ordering 违规者是真实完成,不再静默剔除,
            # 归一化分母随之去偏);旧 ordering-ok 口径经下面两个披露列与
            # completed_ordering_violations 计数保留可追溯。
            "completed_ordering_violations": summary.get(
                "completed_ordering_violations", ""
            ),
            "mean_e2e_ns": summary.get("mean_e2e_ns"),
            "p50_e2e_ns": summary.get("p50_e2e_ns"),
            "p95_e2e_ns": summary.get("p95_e2e_ns"),
            "p99_e2e_ns": summary.get("p99_e2e_ns"),
            "completed_e2e_ordering_ok_mean_ns": summary.get(
                "completed_e2e_ordering_ok_mean_ns"
            ),
            "completed_e2e_ordering_ok_p99_ns": summary.get(
                "completed_e2e_ordering_ok_p99_ns"
            ),
            "drain_tput_rps": summary.get("drain_tput_rps"),
            "sim_window_tput_rps": summary.get("sim_window_tput_rps"),
            "memory_actions_total": summary.get("memory_actions_total", ""),
            "memory_actions_replayed": summary.get("memory_actions_replayed", ""),
            "memory_actions_unresolved": summary.get("memory_actions_unresolved", ""),
        }
    )
    return [row]


# 第一期（die 后端）：die 利用率三元组的管线级断言。C++ MetricCollector
# 在每条 type=rank_compute 记录上输出 active_kernel_die_{compute,memory,
# vector}_util（门控=compute-cost-model=die；roofline 键保持 legacy 门控
# 不动）。本断言在后处理侧 fail-closed：声明 die 后端的 run，每个 rank
# 的三个键必须非 null、数值、且落在 [0, 1+eps]（eps 与 C++
# kUtilEpsilon 同值）。
DIE_UTILIZATION_KEYS = (
    "active_kernel_die_compute_util",
    "active_kernel_die_memory_util",
    "active_kernel_die_vector_util",
)
# 与 C++ MetricCollector 的 kUtilEpsilon 一致（上行在 1e-6 容差内钳制）。
DIE_UTILIZATION_EPSILON = 1e-6
# null 的唯一合法理由：窗口内没有任何计算段（C++ 端 note 原文，与本仓
# roofline 键的 M-17 镜像口径同文案）。
_DIE_NULL_NOTE_OK = "unavailable: zero comp time in utilization window"


def _run_declares_die_backend(run: Run) -> bool:
    """该 run 是否声明 die 计算后端。

    声明源优先级：metrics_manifest.json 的 compute-cost-model（第一期
    起由 plan_materializer 同步写入，归档 run 目录自带）> 同目录
    manifest.json 的同名键（服务 manifest）> 无声明时从 rank_compute
    记录自身推断（C++ 的 note 文案区分 not-die）。"""
    metrics_manifest, service_manifest, _roofline = _load_manifest_sidecars(run)
    for manifest in (metrics_manifest, service_manifest):
        value = manifest.get("compute-cost-model") if isinstance(manifest, dict) else None
        if isinstance(value, str) and value:
            return value == "die"
    for record in run.rank_compute:
        if record.get("active_kernel_die_note") == (
                "unavailable: compute-cost-model is not die"):
            return False
        if any(record.get(key) is not None for key in DIE_UTILIZATION_KEYS):
            return True
    return False


def _check_die_utilization_records(runs) -> None:
    """管线断言（第一期）：die 后端 run 的利用率三元组。

    声明 die 计算后端的 run 必须在每条 rank_compute 记录上携带三个
    active_kernel_die_* 利用率（非 null 数值且 ∈[0,1+eps]）；null 仅在
    携带采集器"零计算窗口"note 时合法。其余情形（缺键、未知 note 的
    null、越界/非数值、声明 die 却无 rank_compute 记录）一律
    fail-closed。legacy run 不受断言（无声明即跳过）。"""
    for run in runs:
        if not _run_declares_die_backend(run):
            continue
        if not run.rank_compute:
            raise PostprocessError(
                f"run {run.run_id}: declares the die compute backend but "
                "carries no type=rank_compute records (the utilization "
                "triplet cannot be verified; truncated log, fail-closed)"
            )
        for index, record in enumerate(run.rank_compute):
            rank = record.get("rank", "?")
            for key in DIE_UTILIZATION_KEYS:
                value = record.get(key)
                if value is None:
                    note = record.get("active_kernel_die_note")
                    if note == _DIE_NULL_NOTE_OK:
                        continue
                    raise PostprocessError(
                        f"run {run.run_id}: rank_compute #{index} "
                        f"(rank {rank}) has {key}=null with note={note!r} "
                        "(die backend utilization must be present on every "
                        "rank; only the zero-comp-window note is a legal "
                        "null)"
                    )
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise PostprocessError(
                        f"run {run.run_id}: rank_compute #{index} "
                        f"(rank {rank}) has non-numeric {key}={value!r}"
                    )
                if not (-DIE_UTILIZATION_EPSILON <= value
                        <= 1.0 + DIE_UTILIZATION_EPSILON):
                    raise PostprocessError(
                        f"run {run.run_id}: rank_compute #{index} "
                        f"(rank {rank}) has {key}={value!r} outside "
                        f"[0, 1+{DIE_UTILIZATION_EPSILON}]"
                    )


def _strict_rank_compute_int(value: Any, field_name: str) -> int:
    """rank_compute 聚合字段的严格整型（真 int 拒 bool，或纯十进制串）。"""
    if isinstance(value, bool):
        raise PostprocessError(
            f"rank_compute {field_name} must be an integer, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isascii() and value.isdigit():
        return int(value, 10)
    raise PostprocessError(
        f"rank_compute {field_name} must be an integer (or its decimal "
        f"string form), got {value!r}")


def _microbench_rows(
    run: Run, labels: dict[str, str], metrics_manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    point = metrics_manifest.get("microbenchmark", {})
    active_ranks = set(point.get("ranks", []))
    ops_by_rank: dict[int, int] = {}
    bytes_by_rank: dict[int, int] = {}
    peak_flops: Optional[float] = None
    local_bw: Optional[float] = None
    for record in run.rank_compute:
        rank = record.get("rank")
        if active_ranks and rank not in active_ranks:
            continue
        # (2026-10-01 审查修复/R2 补正/R3 规整) R9-F1 同款严格整型口径：
        # C++ 以 u128_to_string 落这两个字段——接受真 int（拒 bool）或纯
        # 十进制字符串（精确解析）；拒 float/null/其它（静默截断与裸
        # TypeError 均禁止）。助手定义提前到循环外（R3：原先每迭代重定义）。
        ops_by_rank[rank] = _strict_rank_compute_int(
            record.get("total_num_ops", 0), "total_num_ops")
        bytes_by_rank[rank] = _strict_rank_compute_int(
            record.get("total_local_bytes", 0), "total_local_bytes")
        peak_flops = record.get("peak_flops_per_second") or peak_flops
        local_bw = record.get("local_hbm_bw_bytes_per_second") or local_bw
    rows = []
    implausible_iteration_rows = 0
    for iteration in run.iterations:
        row = _base_row(run, labels, SOURCE_MICROBENCHMARK)
        # L1(face-LRU 审计,2026-09-29):负值/超阈值(uint64 回绕伪时长)的
        # iteration_time_ns 按既有容错风格置空并计数告警,不再原样落盘、
        # 不再进 sec.8.7 重算。
        iteration_time_ns, implausible = _sanity_iteration_time_ns(
            iteration.get("iteration_time_ns")
        )
        if implausible:
            implausible_iteration_rows += 1
        total_ops = sum(ops_by_rank.values())
        total_bytes = sum(bytes_by_rank.values())
        tp_degree = point.get("tp_degree", "")
        compute_util: Optional[float] = None
        hbm_bw_util: Optional[float] = None
        if (
            iteration_time_ns
            and tp_degree
            and peak_flops
            and local_bw
        ):
            window_s = iteration_time_ns / 1e9
            # doc sec.8.7 (derived from rank_compute + iteration records; the
            # C++ iteration record itself does not carry utilization fields).
            compute_util = total_ops / (tp_degree * peak_flops * window_s)
            hbm_bw_util = total_bytes / (tp_degree * local_bw * window_s)
        row.update(
            {
                "benchmark_point_id": iteration.get("benchmark_point_id", ""),
                "phase": point.get("phase", ""),
                "tp_degree": tp_degree,
                "prefill_chunk": point.get("prefill_chunk", ""),
                "decode_batch": point.get("decode_batch", ""),
                "kv_length": point.get("kv_length", ""),
                "repeat_index": point.get("repeat_index", ""),
                "iteration_start_ns": iteration.get("iteration_start_ns"),
                "iteration_end_ns": iteration.get("iteration_end_ns"),
                "iteration_time_ns": iteration_time_ns,
                "total_num_ops": str(total_ops),
                "total_local_bytes": str(total_bytes),
                "compute_util": compute_util,
                "hbm_bw_util": hbm_bw_util,
            }
        )
        rows.append(row)
    if implausible_iteration_rows:
        print(
            f"warning: run {run.run_id}: "
            f"{implausible_iteration_rows} iteration record(s) carried an "
            "implausible iteration_time_ns (negative or above the "
            f"{MAX_PLAUSIBLE_ITERATION_TIME_NS} ns uint64-wrap guard); "
            "field emptied and sec.8.7 recompute suppressed",
            file=sys.stderr,
        )
    return rows


def _planner_roofline_rows(run: Run, labels: dict[str, str], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        row = _base_row(run, labels, SOURCE_PLANNER_ROOFLINE)
        row.update(
            {
                "phase": record.get("phase", ""),
                "tp_degree": record.get("tp_degree", ""),
                "batch": record.get("batch", ""),
                "kv_bin": record.get("kv_bin", ""),
                "prefill_chunk": record.get("prefill_chunk_tokens", ""),
                "count": record.get("count", ""),
                "sum_iteration_time_ns": record.get("sum_iteration_time_ns", ""),
                "min_iteration_time_ns": record.get("min_iteration_time_ns", ""),
                "max_iteration_time_ns": record.get("max_iteration_time_ns", ""),
            }
        )
        rows.append(row)
    return rows


def _memory_ledger_rows(run: Run, labels: dict[str, str]) -> list[dict[str, Any]]:
    peaks_by_rank = {
        record.get("rank"): record for record in run.planner_peaks
    }
    rows = []
    for record in run.capacity_timeavg:
        row = _base_row(run, labels, SOURCE_PLANNER_MEMORY)
        rank = record.get("rank")
        row.update(
            {
                "rank": rank,
                "capacity_bytes": record.get("capacity_bytes"),
                "resident_capacity_timeavg_util": record.get(
                    "resident_capacity_timeavg_util"
                ),
                "committed_capacity_timeavg_util": record.get(
                    "committed_capacity_timeavg_util"
                ),
                "memory_actions_total": record.get("actions_total", ""),
                "memory_actions_replayed": record.get("actions_replayed", ""),
                "memory_actions_unresolved": record.get("actions_unresolved", ""),
            }
        )
        peak = peaks_by_rank.get(rank)
        if peak is not None:
            row["peak_physical_used_bytes"] = peak.get("peak_physical", {}).get(
                "peak_value_bytes", ""
            )
            row["peak_committed_used_bytes"] = peak.get("peak_committed", {}).get(
                "peak_value_bytes", ""
            )
        rows.append(row)
    if not rows:
        # Peaks without replay (e.g. microbenchmark runs) still surface.
        for rank, peak in sorted(peaks_by_rank.items()):
            row = _base_row(run, labels, SOURCE_PLANNER_MEMORY)
            row.update(
                {
                    "rank": rank,
                    "capacity_bytes": peak.get("ledger", {}).get("capacity_bytes", ""),
                    "peak_physical_used_bytes": peak.get("peak_physical", {}).get(
                        "peak_value_bytes", ""
                    ),
                    "peak_committed_used_bytes": peak.get("peak_committed", {}).get(
                        "peak_value_bytes", ""
                    ),
                }
            )
            rows.append(row)
    return rows


def _int_or_none(value: Any) -> Optional[int]:
    """R9-F1（2026-09-29 face 移植）：指标数值严格整型口径——仅接受真
    int（bool 拒）；None/float/str 等一律 None（NA 口径），不再裸 int()
    静默截断（5.7→5）或强转（"5"→5）——错型漂移绝不改写数值。"""
    if value is None or isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _request_incomplete_reason(record: dict[str, Any]) -> str:
    missing = [
        field
        for field in (
            "arrival_ns",
            "prefill_start_ns",
            "prefill_end_ns",
            "decode_start_ns",
            "completion_ns",
        )
        if record.get(field) is None
    ]
    return "incomplete_reason=missing:" + (",".join(missing) if missing else "unknown")


# ---------------------------------------------------------------------------
# WP9 fallback (B4, 2026-08-26): train-interpolated first-token proxy.
#
# B3_S3 60s gate-2 (decision equivalence) failed -- the split's physical
# perturbation is amplified by the closed loop (see
# /tmp/slo_wps/gates/B3_S3.FAILED, evidence /tmp/slo_wps/b3/S3/) -- so the
# default first-token source falls back to the spec 1.6 proxy
# (SH_FIRST_TOKEN_SPLIT now defaults to "0"):
#
#   first_token_proxy = decode_start_ns
#                     + (w1 / sum_i w_i) * (first_train_end_ns - decode_start_ns)
#   w_i = W_bytes + KV_bytes(context + consumed + i)     [i = 1..P, 1-based]
#
# with the weight sum running over the train's iterations (i = 1..N,
# N = the ledger row's train length).  The w_i mirror the scheduler's own
# pass-span encoding for a train member (sh30_online_scheduler.py
# _plan_train: span step i is (1, context + consumed + i)): the debut's
# first token completes with the train's FIRST iteration, so the
# interpolation weight is iteration 1's weight over the train's N
# iteration weights.  The debut joins decode exactly once with consumed=0
# (sticky decode; verified 1454/1454 joiner entries == requests in the 60s
# reference run), hence consumed=0.
#
# first_train_end_ns: the train ledger row's tick is the emission (start)
# boundary; an instance is busy until the train's end barrier completes
# (one-train-in-flight gate), so the NEXT ledger row on the same instance
# is the first decision boundary after the end barrier and is used as the
# train-end tick (last train of an instance -> proxy not obtainable -> NA).
# Degenerate 1-iteration trains (N=1 -> share=1) make the proxy equal that
# next-emission boundary, which sits after the barrier tick recorded as
# completion; such values are clamped to completion (the first token
# physically cannot complete after the request) and flagged
# "clamped_to_completion" in instructions (7/1454 rows in the 60s
# reference run).
#
# W_bytes / KV bytes per token: the repository's authoritative conversions
# (face_scheduler.estimate_model_weight_bytes -- weights are read once per
# iteration -- and kv_cache_bytes_for_tokens), parameterised by the
# trace_config.csv model rows; no value is invented here.
#
# The proxy is a display metric ONLY: first_token_source=train_interpolated
# is rejected from every SLO judgment path (slo_common
# assert_no_proxy_columns; unit-tested in slo_tools/tests).
# ---------------------------------------------------------------------------

# The proxy reads the run's ARCHIVED audit ledger (results/train_ledger.
# jsonl, moved there by the runner after its in-run postprocess step): the
# in-run request_metrics.csv therefore stays NA when the split is off (no
# exact events), and re-running the postprocessor over an archived run dir
# fills the proxy.  This keeps run products and offline analysis distinct.
TRAIN_LEDGER_CANDIDATES = (
    Path("results") / "train_ledger.jsonl",
)

_FIRST_TOKEN_SOURCE_EXACT = "exact"
_FIRST_TOKEN_SOURCE_PROXY = "train_interpolated"

# G5.5（2026-09-27）contract constants: the closed enumeration surfaces of
# request_metrics.csv are pinned by tests/test_metrics_contract.py against
# THESE module-level names (not against the test's own copy), so the
# producer and the contract test share one source of truth.  In-row writers
# below reference the names -- zero behavior change, literals deduplicated.
_TERMINAL_STATUS_COMPLETED = "completed"
_TERMINAL_STATUS_FAILED = "failed"
_REQUEST_STATUS_NOT_ARRIVED = "not_arrived"
_REQUEST_STATUS_IN_FLIGHT = "in_flight"
TERMINAL_STATUS_VALUES = (
    _TERMINAL_STATUS_COMPLETED, _TERMINAL_STATUS_FAILED,
    _REQUEST_STATUS_NOT_ARRIVED, _REQUEST_STATUS_IN_FLIGHT)
NA_SENTINEL = "NA"
FIRST_TOKEN_SOURCE_VALUES = (
    _FIRST_TOKEN_SOURCE_EXACT,
    _FIRST_TOKEN_SOURCE_PROXY,
    NA_SENTINEL,
)


# L1(face-LRU 审计,2026-09-29):iteration_time_ns 合理性上限(10^15 ns ≈
# 11.6 天,远高于任何真实 iteration,远低于 uint64 回绕产生的 ~1.8e19 伪
# 时长)。C++ 端自同日起对 end<start 窗口 clamp 为 0 并在 consistency 记录
# 计数(iteration_time_underflow_clamped);旧日志仍可能携带回绕值/负值,
# 按本工具既有容错风格跳过并打 stderr 警告(带计数),不伪造值。
MAX_PLAUSIBLE_ITERATION_TIME_NS = 10 ** 15


def _sanity_iteration_time_ns(value: Any) -> tuple[Optional[int], bool]:
    """(value_or_None, implausible_flag) for one iteration_time_ns.

    (None, False) = field absent; (None, True) = present but implausible
    (negative or above the uint64-wrap guard threshold); (parsed, False) =
    usable. Never raises, never fabricates.
    """
    if value is None:
        return None, False
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None, True
    if 0 <= parsed <= MAX_PLAUSIBLE_ITERATION_TIME_NS:
        return parsed, False
    return None, True


def _load_model_bytes() -> tuple[Optional[int], Optional[int]]:
    """(W_bytes, kv_bytes_per_token) from this workload's trace_config.csv.

    Returns (None, None) when the config or a model row is missing/invalid
    (proxy then degrades to NA; nothing is fabricated).
    """

    try:
        from face_scheduler import (
            FaceModel,
            estimate_model_weight_bytes,
            kv_cache_bytes_for_tokens,
        )
    except ImportError:
        return None, None
    config_path = MODULE_DIR / "trace_config.csv"
    values: dict[str, str] = {}
    try:
        with config_path.open(encoding="utf-8") as source:
            reader = csv.DictReader(source)
            for row in reader:
                if row.get("kind") == "config":
                    key = (row.get("key") or "").strip()
                    if key:
                        values[key] = (row.get("value") or "").strip()
    except OSError:
        return None, None

    def _positive_int(key: str) -> Optional[int]:
        raw = values.get(key)
        try:
            parsed = int(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    model_kwargs: dict[str, Any] = {}
    for key in ("layers", "hidden_size", "ffn_size", "num_heads",
                "vocab_size", "bytes_per_elem"):
        parsed = _positive_int(key)
        if parsed is None:
            return None, None
        model_kwargs[key] = parsed
    model_kwargs["mlp_variant"] = values.get("mlp_variant") or "gelu"
    try:
        model = FaceModel(**model_kwargs)
        return estimate_model_weight_bytes(model), kv_cache_bytes_for_tokens(model, 1)
    except ValueError:
        return None, None


class _TrainProxyIndex:
    """Lazy per-run index over results/train_ledger.jsonl (read-only)."""

    def __init__(self, run: Run):
        self._run = run
        self._loaded = False
        # request_id -> (ledger_row_position, row) of its first (non
        # first_step) train carrying it in joiners.
        self.first_train_by_request: dict[str, tuple[int, dict[str, Any]]] = {}
        # ledger row position -> tick of the next row on the same instance.
        self.next_tick_by_position: dict[int, int] = {}
        # R9-A2（2026-09-29 face 移植）：ledger 在场却不可读时的丢弃原因
        # （"文件名(异常类型)"）；None = 单纯缺失。下游降级 note 据此区分
        # "缺失"与"损坏丢弃"——旧写法两者同词 no_train_ledger，审计员被
        # 误导为未归档。
        self.load_failure: Optional[str] = None

    def available(self) -> bool:
        self._ensure_loaded()
        return self._available

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        self._available = False
        ledger_path: Optional[Path] = None
        for candidate in TRAIN_LEDGER_CANDIDATES:
            path = self._run.log_path.parent / candidate
            if path.is_file():
                ledger_path = path
                break
        if ledger_path is None:
            return
        rows: list[dict[str, Any]] = []
        try:
            with ledger_path.open(encoding="utf-8") as source:
                for line in source:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if isinstance(record, dict):
                        rows.append(record)
        except (OSError, json.JSONDecodeError) as exc:
            # R9-A2（2026-09-29 face 移植）：在场却损坏 ≠ 缺失——记丢弃
            # 原因供降级 note 区分（整册丢弃的降级语义本身保留：proxy 是
            # display-only fallback，损坏不阻断后处理）。
            self.load_failure = "{} ({})".format(
                ledger_path.name, type(exc).__name__)
            return
        for position, row in enumerate(rows):
            if row.get("first_step"):
                # WP9 split-ON two-phase emission: the first_step row is a
                # metrics-only pre-row; the remainder row of the same train
                # carries the canonical joiners record.
                continue
            for request_id in row.get("joiners") or []:
                if isinstance(request_id, str) and request_id not in (
                        self.first_train_by_request):
                    self.first_train_by_request[request_id] = (position, row)
        next_by_instance: dict[Any, int] = {}
        for position in range(len(rows) - 1, -1, -1):
            instance = rows[position].get("instance_index")
            if instance in next_by_instance:
                self.next_tick_by_position[position] = next_by_instance[instance]
            tick = rows[position].get("tick")
            if isinstance(tick, int):
                next_by_instance[instance] = tick
        self._available = True


_PROXY_MODEL_BYTES: Optional[tuple[Optional[int], Optional[int]]] = None


def _model_bytes_cached() -> tuple[Optional[int], Optional[int]]:
    global _PROXY_MODEL_BYTES
    if _PROXY_MODEL_BYTES is None:
        _PROXY_MODEL_BYTES = _load_model_bytes()
    return _PROXY_MODEL_BYTES


def _first_token_proxy_value(
    index: _TrainProxyIndex,
    entry: dict[str, Any],
    record: dict[str, Any],
) -> tuple[Optional[int], str]:
    """(proxy_ns, note) for one request; (None, reason-note) when the proxy
    is not obtainable from the archived artifacts."""

    request_id = str(entry.get("request_id") or "")
    decode_start = _int_or_none(record.get("decode_start_ns"))
    if decode_start is None:
        return None, "proxy_unavailable:decode_start_ns"
    found = index.first_train_by_request.get(request_id)
    if found is None:
        return None, "proxy_unavailable:no_train_ledger_joiner_row"
    position, row = found
    iterations = _int_or_none(row.get("iterations"))
    if iterations is None or iterations <= 0:
        return None, "proxy_unavailable:train_iterations"
    train_end = index.next_tick_by_position.get(position)
    if train_end is None:
        return None, "proxy_unavailable:no_next_train_on_instance(last_train)"
    if train_end < decode_start:
        return None, "proxy_unavailable:train_end<decode_start"
    # context at decode start: prefill_context_tokens (history + folded
    # prefill); fall back to the equivalent derivations.
    decode_length = _int_or_none(entry.get("decode_length"))
    context = _int_or_none(entry.get("prefill_context_tokens"))
    if context is None:
        history = _int_or_none(entry.get("history_tokens_before")) or 0
        prefill = _int_or_none(entry.get("prefill_length"))
        if prefill is not None:
            context = history + prefill
    if context is None and decode_length is not None:
        final = _int_or_none(entry.get("final_context_tokens"))
        if final is not None:
            context = final - decode_length
    if context is None or context < 0:
        return None, "proxy_unavailable:context_tokens"
    weight_bytes, kv_per_token = _model_bytes_cached()
    if not weight_bytes or not kv_per_token:
        return None, "proxy_unavailable:W_bytes/KV_bytes(trace_config)"
    # debut join: consumed = 0.  The weight sum runs over the TRAIN's
    # iterations (i = 1..iterations, the ledger row's train length): the
    # debut's first token completes with the train's FIRST iteration, and
    # w_i linearises the per-iteration cost growth as the member KV grows
    # (i is the iteration ordinal, mirroring the scheduler's pass-span
    # encoding (1, context + consumed + step)).  Summing over the debut's
    # own participation instead would degenerate to share=1 whenever the
    # debut exits before the train boundary (decode_length < iterations)
    # and overshoot completion (empirically 7/1454 rows in the 60s
    # reference run).
    consumed = 0
    iteration_count = iterations
    w_first = weight_bytes + kv_per_token * (context + consumed + 1)
    weight_total = sum(
        weight_bytes + kv_per_token * (context + consumed + step)
        for step in range(1, iteration_count + 1)
    )
    share = w_first / weight_total
    proxy_ns = int(round(decode_start + share * (train_end - decode_start)))
    participation = (
        min(decode_length, iterations)
        if decode_length is not None else None)
    note = (
        "first_token: proxy(train_interpolated) "
        f"decode_start={decode_start} first_train_end={train_end} "
        f"train={row.get('train_id')} inst={row.get('instance_index')} "
        f"N={iteration_count} P={participation} ctx={context} "
        f"W_bytes={weight_bytes} kv_per_token={kv_per_token}"
    )
    # Physical upper bound: a request's first token cannot complete after
    # the request itself.  Degenerate 1-iteration trains (N=1 -> share=1)
    # make the proxy equal the next-emission boundary, which sits AFTER the
    # end-barrier tick recorded as completion (delivery+scheduling gap,
    # 3-65ms in the 60s reference run, 7/1454 rows).  Clamp to completion
    # and say so -- for decode_length==1 the exact-mode semantics is
    # first_token==completion anyway.
    completion = _int_or_none(record.get("completion_ns"))
    if completion is not None and proxy_ns > completion:
        proxy_ns = completion
        note += " clamped_to_completion(train_end_boundary>completion)"
    elif (completion is not None and decode_length == 1
          and proxy_ns < completion):
        # decode_length==1: the request's only decode iteration IS its first
        # token, so exact-mode semantics is first_token==completion (see the
        # clamp note above).  A dl1 debut riding a MULTI-iteration train gets
        # share<1 and the interpolation lands BEFORE the recorded completion
        # tick (face 0902 full_rt: 18/268 dl1 rows, up to 4.9s early); pin to
        # completion -- the same value the N=1 clamp above yields.
        proxy_ns = completion
        note += " pinned_to_completion(decode_length==1,proxy<completion)"
    return proxy_ns, note


def _request_rows_for_run(run: Run) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Per-request rows for one full-detail service run (B1/WP1).

    Joins the log's ``request`` records against the ``manifest.json``
    sidecar (sibling of the init record's ``manifest_path``) by
    queue_index/request_id.  Any join failure, row-count mismatch, or
    timing-invariant violation is a sec.11.4-style fail-closed abort.
    """

    _, service_manifest, _ = _load_manifest_sidecars(run)
    entries = service_manifest.get("requests") if isinstance(service_manifest, dict) else None
    if not isinstance(entries, list) or not entries:
        raise PostprocessError(
            f"run {run.run_id}: request records require a manifest.json "
            "sidecar with a non-empty requests[] list (manifest_path="
            f"{run.init.get('manifest_path')!r})"
        )
    by_queue_index: dict[int, dict[str, Any]] = {}
    for entry in entries:
        queue_index = _int_or_none(entry.get("queue_index"))
        if queue_index is None:
            raise PostprocessError(
                f"run {run.run_id}: manifest.json entry without a valid "
                f"queue_index: {json.dumps(entry, sort_keys=True)[:120]}"
            )
        if queue_index in by_queue_index:
            raise PostprocessError(
                f"run {run.run_id}: duplicate queue_index {queue_index} in "
                "manifest.json requests[]"
            )
        by_queue_index[queue_index] = entry
    if len(entries) != len(run.requests):
        raise PostprocessError(
            f"run {run.run_id}: row-count mismatch between manifest.json "
            f"requests[] ({len(entries)}) and request records "
            f"({len(run.requests)})"
        )

    type_counts = {"human": 0, "tool": 0, "unknown": 0}
    rows: list[dict[str, Any]] = []
    # WP9 fallback (B4): train-interpolated proxy index over the run's
    # train_ledger.jsonl (loaded lazily; absent ledger -> NA with a note).
    proxy_index = _TrainProxyIndex(run)
    proxy_available = proxy_index.available()
    for record in sorted(run.requests, key=lambda item: item.get("queue_index") or 0):
        queue_index = _int_or_none(record.get("queue_index"))
        entry = by_queue_index.get(queue_index) if queue_index is not None else None
        if entry is None:
            raise PostprocessError(
                f"run {run.run_id}: request record queue_index "
                f"{queue_index!r} has no manifest.json entry (join failure)"
            )
        if record.get("request_id") != entry.get("request_id"):
            raise PostprocessError(
                f"run {run.run_id}: queue_index {queue_index} request_id "
                f"mismatch: log {record.get('request_id')!r} vs manifest "
                f"{entry.get('request_id')!r}"
            )
        for field in ("session_id", "turn_index"):
            if record.get(field) != entry.get(field):
                raise PostprocessError(
                    f"run {run.run_id}: queue_index {queue_index} {field} "
                    f"mismatch: log {record.get(field)!r} vs manifest "
                    f"{entry.get(field)!r}"
                )

        completed = record.get("completed") is True
        # A stopped observation window does not establish request failure.
        # The metrics stream contains the entire planned cohort, including
        # closed-loop successors whose arrival has not yet been released.
        if completed:
            request_status = _TERMINAL_STATUS_COMPLETED
        elif record.get("terminal_status") == _TERMINAL_STATUS_FAILED:
            request_status = _TERMINAL_STATUS_FAILED
        elif _int_or_none(record.get("arrival_ns")) is None:
            request_status = _REQUEST_STATUS_NOT_ARRIVED
        else:
            request_status = _REQUEST_STATUS_IN_FLIGHT
        notes: list[str] = []
        # WP9（2026-08-26）：首 token 事件值与 C++ 侧 note（withheld 原因
        # 留痕；decode_length==1 的 min/max 聚合不对称放宽裁决见
        # WP9_CONTRACT，B2.5 在 C++ 侧统一修）。
        first_token_ns = _int_or_none(record.get("first_token_ns"))
        first_token_source = _FIRST_TOKEN_SOURCE_EXACT if (
            first_token_ns is not None) else None
        first_token_note = record.get("first_token_note")
        if isinstance(first_token_note, str) and first_token_note:
            notes.append(f"first_token: {first_token_note}")
        if first_token_ns is None and proxy_available:
            # WP9 fallback (B4): no exact code-8 value -> train-interpolated
            # proxy when the archived train ledger yields the debut request's
            # first-train boundary; otherwise NA (note records the reason).
            proxy_ns, proxy_note = _first_token_proxy_value(
                proxy_index, entry, record)
            if proxy_ns is not None:
                first_token_ns = proxy_ns
                first_token_source = _FIRST_TOKEN_SOURCE_PROXY
            notes.append(proxy_note)
        elif first_token_ns is None:
            if proxy_index.load_failure is not None:
                # R9-A2（2026-09-29 face 移植）：ledger 在场却损坏——降级
                # note 区分于缺失（no_train_ledger），不再误导审计员。
                notes.append(
                    "proxy_unavailable:train_ledger_corrupt({})".format(
                        proxy_index.load_failure))
            else:
                notes.append("proxy_unavailable:no_train_ledger(results/)")
        if first_token_source is None:
            first_token_source = NA_SENTINEL
        request_type = entry.get("request_type")
        if request_type in REQUEST_TYPE_KNOWN:
            type_counts[request_type] += 1
        else:
            request_type = "unknown"
            type_counts["unknown"] += 1
            notes.append(
                "request_type=unknown (no human/tool value in manifest.json)"
            )
        if not completed:
            notes.append(_request_incomplete_reason(record))

        arrival = _int_or_none(record.get("arrival_ns"))
        completion = _int_or_none(record.get("completion_ns"))
        e2e = _int_or_none(record.get("e2e_ns"))
        queue_ns = _int_or_none(record.get("queue_ns"))
        prefill_ns = _int_or_none(record.get("prefill_ns"))
        gap_ns = _int_or_none(record.get("prefill_decode_gap_ns"))
        decode_ns = _int_or_none(record.get("decode_ns"))
        # Invariants on rows that carry the values (sec.12.6 re-check,
        # fail-closed): arrival <= completion, e2e = completion - arrival,
        # stage decomposition sums to e2e.
        if arrival is not None and completion is not None and arrival > completion:
            raise PostprocessError(
                f"run {run.run_id}: queue_index {queue_index} violates "
                f"arrival<=completion ({arrival} > {completion})"
            )
        if (
            e2e is not None
            and arrival is not None
            and completion is not None
            and e2e != completion - arrival
        ):
            raise PostprocessError(
                f"run {run.run_id}: queue_index {queue_index} violates "
                f"e2e=completion-arrival ({e2e} != {completion} - {arrival})"
            )
        if (
            e2e is not None
            and queue_ns is not None
            and prefill_ns is not None
            and gap_ns is not None
            and decode_ns is not None
            and queue_ns + prefill_ns + gap_ns + decode_ns != e2e
        ):
            raise PostprocessError(
                f"run {run.run_id}: queue_index {queue_index} violates "
                "queue+prefill+gap+decode=e2e "
                f"({queue_ns}+{prefill_ns}+{gap_ns}+{decode_ns} != {e2e})"
            )
        # dl1 hard invariant (proxy rows only): a decode_length==1 proxy row
        # MUST equal completion -- the pin/clamp branches above guarantee it;
        # exact rows are exempt (WP9_CONTRACT sec.6 TP-skew: first_token may
        # legitimately sit at/before completion).
        if (first_token_source == _FIRST_TOKEN_SOURCE_PROXY
                and first_token_ns is not None and completion is not None
                and _int_or_none(entry.get("decode_length")) == 1
                and first_token_ns != completion):
            raise PostprocessError(
                f"run {run.run_id}: queue_index {queue_index} violates "
                "decode_length==1 proxy first_token_ns == completion_ns "
                "(dl1 first token IS the last token; "
                "pinned_to_completion path broken)"
            )

        row = {
            "queue_index": queue_index,
            "request_id": record.get("request_id", ""),
            "session_id": record.get("session_id", ""),
            "turn_index": record.get("turn_index", ""),
            "request_type": request_type,
            "terminal_status": request_status,
            "arrival_ns": arrival,
            "prefill_start_ns": _int_or_none(record.get("prefill_start_ns")),
            "prefill_end_ns": _int_or_none(record.get("prefill_end_ns")),
            "decode_start_ns": _int_or_none(record.get("decode_start_ns")),
            # WP9: exact code-8 value when the split ran; otherwise the
            # train-interpolated proxy (B4 fallback, source tagged; display
            # metric only -- rejected from SLO judgment paths) or NA.
            "first_token_ns": first_token_ns,
            "first_token_source": first_token_source,
            "completion_ns": completion,
            "queue_ns": queue_ns,
            "prefill_ns": prefill_ns,
            "prefill_decode_gap_ns": gap_ns,
            "decode_ns": decode_ns,
            "e2e_ns": e2e,
            "kv_hit_state": NA_SENTINEL,
            "restore_start_ns": NA_SENTINEL,
            "restore_complete_ns": NA_SENTINEL,
            "pre_prefill_restore_ns": NA_SENTINEL,
            "hidden_restore_ns": NA_SENTINEL,
            "exposed_restore_stall_ns": NA_SENTINEL,
            "hidden_ratio": NA_SENTINEL,
            "prefill_length": _int_or_none(entry.get("prefill_length")),
            "decode_length": _int_or_none(entry.get("decode_length")),
            "prefix_len": _int_or_none(entry.get("prefix_len")),
            "instructions": "; ".join(notes),
        }
        rows.append(row)
    return rows, type_counts


def build_request_metrics(
    runs: Sequence[Run],
) -> tuple[list[dict[str, Any]], dict[str, int], list[str]]:
    """Collect request_metrics rows across runs (full-detail service runs)."""

    rows: list[dict[str, Any]] = []
    type_counts = {"human": 0, "tool": 0, "unknown": 0}
    notes: list[str] = []
    for run in runs:
        detail = run.init.get("detail_level", "")
        run_mode = run.init.get("run_mode", "service")
        if run_mode == "microbenchmark":
            if run.requests:
                notes.append(
                    f"run {run.run_id}: skipped (microbenchmark runs carry no "
                    "manifest.json request join)"
                )
            continue
        if detail == "summary":
            notes.append(
                f"run {run.run_id}: no request_metrics rows (detail_level="
                "summary emits no per-request records)"
            )
            continue
        if not run.requests:
            if detail == "full":
                raise PostprocessError(
                    f"run {run.run_id}: detail_level=full but the log carries "
                    "no request records"
                )
            notes.append(
                f"run {run.run_id}: no request_metrics rows (no request "
                "records in the log)"
            )
            continue
        run_rows, run_counts = _request_rows_for_run(run)
        rows.extend(run_rows)
        for key, value in run_counts.items():
            type_counts[key] = type_counts.get(key, 0) + value
    return rows, type_counts, notes


def build_raw_rows(
    runs: Sequence[Run], run_configs: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        _check_request_completion(run)
        metrics_manifest, service_manifest, roofline_sidecar = _load_manifest_sidecars(run)
        labels = _labels(run, service_manifest, run_configs.get(run.run_id, {}))
        run_mode = run.init.get("run_mode", "service")
        if run_mode == "microbenchmark":
            rows.extend(_microbench_rows(run, labels, metrics_manifest))
        else:
            rows.extend(_service_rows(run, labels))
        rows.extend(
            _planner_roofline_rows(
                run, labels, run.planner_roofline or roofline_sidecar
            )
        )
        rows.extend(_memory_ledger_rows(run, labels))
    return rows


@dataclass
class Group:
    group_id: str
    runs: list[Run]
    allow_digest_mismatch: bool
    baseline_run_id: Optional[str]
    labels_by_run: dict[str, dict[str, Any]]


def build_groups(
    runs: Sequence[Run], group_config: Optional[dict[str, Any]]
) -> tuple[list[Group], dict[str, dict[str, Any]]]:
    """Comparison groups; returns (groups, per-run label overrides)."""

    run_configs: dict[str, dict[str, Any]] = {}
    by_id = {run.run_id: run for run in runs}
    groups: list[Group] = []
    if group_config is None:
        # Default: automatic homogeneous groups keyed by workload identity, so
        # mixed feeds never silently mix different workloads (sec.11.4).
        auto: dict[tuple[Any, ...], list[Run]] = {}
        for run in runs:
            key = (
                run.init.get("repo_variant", ""),
                run.init.get("run_mode", ""),
                run.init.get("trace_digest", ""),
                run.init.get("request_mapping_digest", ""),
            )
            auto.setdefault(key, []).append(run)
        for index, members in enumerate(
            sorted(auto.values(), key=lambda members: members[0].run_id)
        ):
            groups.append(
                Group(
                    group_id=f"auto_{index}",
                    runs=members,
                    allow_digest_mismatch=False,
                    baseline_run_id=None,
                    labels_by_run={},
                )
            )
        return groups, run_configs

    # Report 3.6 fix (2026-09-29, sister-repo sync from astra-sim-wscllm): a
    # --group-config JSON value of the wrong shape (a top-level
    # array/string/number, a non-object group entry, a non-object run entry)
    # used to escape the chained .get calls below as a bare AttributeError --
    # outside the PostprocessError channel that main() is the only catcher
    # of.  The group config drives comparison-group keys, so every level is
    # validated as the object shape its access pattern requires and a wrong
    # shape aborts fail-closed (sec.11.4 family).
    if not isinstance(group_config, dict):
        raise PostprocessError(
            "group config must be a JSON object with a 'groups' list "
            f"(got {type(group_config).__name__} {group_config!r}; "
            "fail-closed)"
        )
    configured_groups = group_config.get("groups")
    if not isinstance(configured_groups, list) or not configured_groups:
        raise PostprocessError("group config must contain a non-empty 'groups' list")
    for entry_index, entry in enumerate(configured_groups):
        if not isinstance(entry, dict):
            raise PostprocessError(
                f"group config entries[{entry_index}] must be an object "
                f"(got {type(entry).__name__} {entry!r}; fail-closed)"
            )
        group_id = entry.get("group_id")
        if not group_id:
            raise PostprocessError("group config entry without group_id")
        members: list[Run] = []
        labels_by_run: dict[str, dict[str, Any]] = {}
        baseline_run_id = entry.get("baseline_run_id")
        baseline_marks = 0
        group_runs = entry.get("runs", [])
        if not isinstance(group_runs, list):
            raise PostprocessError(
                f"group {group_id}: 'runs' must be a list "
                f"(got {type(group_runs).__name__} {group_runs!r}; "
                "fail-closed)"
            )
        for run_entry in group_runs:
            if not isinstance(run_entry, dict):
                raise PostprocessError(
                    f"group {group_id}: every run entry must be an object "
                    f"(got {type(run_entry).__name__} {run_entry!r}; "
                    "fail-closed)"
                )
            run_id = run_entry.get("run_id")
            run = by_id.get(run_id)
            if run is None:
                raise PostprocessError(
                    f"group {group_id}: run_id {run_id!r} not found in the input logs"
                )
            members.append(run)
            labels = {
                key: run_entry[key]
                for key in ("model", "dataset", "hardware", "kv_policy")
                if key in run_entry
            }
            labels_by_run[run_id] = labels
            run_configs[run_id] = labels
            if run_entry.get("baseline"):
                baseline_marks += 1
                baseline_run_id = run_id
        if baseline_marks > 1:
            raise PostprocessError(
                f"group {group_id}: baseline is not unique (sec.11.4)"
            )
        if not entry.get("allow_digest_mismatch", False):
            digests = {
                (
                    run.init.get("trace_digest", ""),
                    run.init.get("request_mapping_digest", ""),
                )
                for run in members
            }
            if len(digests) > 1:
                raise PostprocessError(
                    f"group {group_id}: workload digest mismatch across runs "
                    "without allow_digest_mismatch (sec.11.4)"
                )
        schemas = set()
        for run in members:
            schemas |= run.schema_versions
        if schemas != {SUPPORTED_SCHEMA}:
            raise PostprocessError(
                f"group {group_id}: mixed metric schema versions "
                f"{sorted(schemas)} must not be normalized together (sec.11.4)"
            )
        groups.append(
            Group(
                group_id=group_id,
                runs=members,
                allow_digest_mismatch=bool(entry.get("allow_digest_mismatch", False)),
                baseline_run_id=baseline_run_id,
                labels_by_run=labels_by_run,
            )
        )
    return groups, run_configs


def _normalize_value(
    *,
    method: str,
    metric_name: str,
    raw_value: float,
    group_id: str,
    group_values: list[float],
    baseline_run_id: Optional[str],
    baseline_value: Optional[float],
) -> Any:
    try:
        return build_normalization_record(
            metric_name,
            raw_value,
            method,
            group_id,
            group_values=group_values,
            baseline_run_id=baseline_run_id,
            baseline_value=baseline_value,
        )
    except MetricsSchemaError as error:
        raise PostprocessError(
            f"group {group_id}: cannot normalize {metric_name} (sec.11.4): {error}"
        ) from None


def build_normalized_rows(
    groups: Sequence[Group], method: str, baseline_cli: Optional[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        baseline_run_id = baseline_cli or group.baseline_run_id
        if method == NORMALIZATION_RATIO_TO_BASELINE:
            matches = [run for run in group.runs if run.run_id == baseline_run_id]
            if baseline_run_id is None or not matches:
                raise PostprocessError(
                    f"group {group.group_id}: ratio_to_baseline requires exactly "
                    "one baseline run; none selected (sec.11.4)"
                )
            if len(matches) > 1:
                raise PostprocessError(
                    f"group {group.group_id}: baseline is not unique (sec.11.4)"
                )
        elif method != NORMALIZATION_GROUP_MAX:
            raise PostprocessError(f"unknown normalization method {method!r}")

        # Service rows: one per run.  Microbenchmark rows: one per point.
        service: list[tuple[Run, float, float]] = []
        microbench: list[tuple[Run, dict[str, Any], float]] = []
        for run in group.runs:
            if run.init.get("run_mode") == "microbenchmark":
                implausible_excluded = 0
                for record in run.iterations:
                    # L1(face-LRU 审计,2026-09-29):implausible 的
                    # iteration_time_ns(负值/uint64 回绕伪时长)不进归一化
                    # 分母,按既有容错风格跳过并计数告警。
                    iteration_time, implausible = _sanity_iteration_time_ns(
                        record.get("iteration_time_ns")
                    )
                    if implausible:
                        implausible_excluded += 1
                        continue
                    if iteration_time:
                        microbench.append((run, record, float(iteration_time)))
                if implausible_excluded:
                    print(
                        f"warning: run {run.run_id}: "
                        f"{implausible_excluded} iteration record(s) "
                        "excluded from the iteration-time normalization "
                        "group (implausible iteration_time_ns guard)",
                        file=sys.stderr,
                    )
            else:
                summary = run.summary
                # M15(face-LRU 审计,2026-09-29):mean_e2e_ns 口径 = 全部
                # completed 请求(ordering 违规者不再被静默剔除),归一化
                # 分母随之去偏;旧 ordering-ok 口径见 summary 的
                # completed_e2e_ordering_ok_* 披露字段。
                mean_e2e = summary.get("mean_e2e_ns")
                drain_tput = summary.get("drain_tput_rps")
                if mean_e2e is None or drain_tput is None:
                    raise PostprocessError(
                        f"run {run.run_id}: summary lacks mean_e2e_ns/"
                        "drain_tput_rps; cannot normalize (sec.11.4)"
                    )
                service.append((run, float(mean_e2e), float(drain_tput)))

        def denominator(values: list[float], baseline_value: Optional[float]) -> float:
            if method == NORMALIZATION_GROUP_MAX:
                if not values:
                    raise PostprocessError(
                        f"group {group.group_id}: empty normalization group "
                        "(sec.11.4)"
                    )
                value = max(values)
            else:
                if baseline_value is None:
                    raise PostprocessError(
                        f"group {group.group_id}: baseline run has no usable "
                        "value for normalization (sec.11.4)"
                    )
                value = baseline_value
            if value <= 0:
                raise PostprocessError(
                    f"group {group.group_id}: normalization denominator is 0 "
                    "(sec.11.4)"
                )
            return value

        baseline_service = (
            next(
                ((mean, tput) for run, mean, tput in service if run.run_id == baseline_run_id),
                None,
            )
            if baseline_run_id
            else None
        )
        time_den = denominator(
            [mean for _, mean, _ in service],
            None if baseline_service is None else baseline_service[0],
        ) if service else None
        tput_den = denominator(
            [tput for _, _, tput in service],
            None if baseline_service is None else baseline_service[1],
        ) if service else None
        baseline_iteration = (
            next(
                (value for run, _, value in microbench if run.run_id == baseline_run_id),
                None,
            )
            if baseline_run_id
            else None
        )
        iteration_den = denominator(
            [value for _, _, value in microbench], baseline_iteration
        ) if microbench else None

        for run, mean_e2e, drain_tput in service:
            time_record = _normalize_value(
                method=method,
                metric_name="mean_e2e_ns",
                raw_value=mean_e2e,
                group_id=group.group_id,
                group_values=[mean for _, mean, _ in service],
                baseline_run_id=baseline_run_id,
                baseline_value=(
                    None if baseline_service is None else baseline_service[0]
                ),
            )
            tput_record = _normalize_value(
                method=method,
                metric_name="drain_tput_rps",
                raw_value=drain_tput,
                group_id=group.group_id,
                group_values=[tput for _, _, tput in service],
                baseline_run_id=baseline_run_id,
                baseline_value=(
                    None if baseline_service is None else baseline_service[1]
                ),
            )
            rows.append(
                {
                    "run_id": run.run_id,
                    "benchmark_point_id": "",
                    "repo_variant": run.init.get("repo_variant", ""),
                    "source": SOURCE_SIMULATOR,
                    "normalization_group_id": group.group_id,
                    "normalization_method": method,
                    "baseline_run_id": baseline_run_id or "",
                    "denominator_time": time_den,
                    "denominator_tput": tput_den,
                    "denominator_iteration_time": "",
                    "normalized_time": time_record.normalized_value,
                    "normalized_tput": tput_record.normalized_value,
                    "normalized_iteration_time": "",
                }
            )
        for run, record, iteration_time in microbench:
            iteration_record = _normalize_value(
                method=method,
                metric_name="iteration_time_ns",
                raw_value=iteration_time,
                group_id=group.group_id,
                group_values=[value for _, _, value in microbench],
                baseline_run_id=baseline_run_id,
                baseline_value=baseline_iteration,
            )
            rows.append(
                {
                    "run_id": run.run_id,
                    "benchmark_point_id": record.get("benchmark_point_id", ""),
                    "repo_variant": run.init.get("repo_variant", ""),
                    "source": SOURCE_MICROBENCHMARK,
                    "normalization_group_id": group.group_id,
                    "normalization_method": method,
                    "baseline_run_id": baseline_run_id or "",
                    "denominator_time": "",
                    "denominator_tput": "",
                    "denominator_iteration_time": iteration_den,
                    "normalized_time": "",
                    "normalized_tput": "",
                    "normalized_iteration_time": iteration_record.normalized_value,
                }
            )
    return rows


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _write_request_csv(
    path: Path, columns: Sequence[str], rows: Sequence[dict[str, Any]]
) -> None:
    """Like _write_csv but renders missing (None) cells as NA."""

    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: (
                        "NA" if row.get(column) is None else row.get(column, "")
                    )
                    for column in columns
                }
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cross-run metrics post-processing (doc sec.11)."
    )
    parser.add_argument("logs", nargs="+", help="run log(s) to parse")
    parser.add_argument("--out-raw", default="raw_metrics.csv")
    parser.add_argument("--out-normalized", default="normalized_metrics.csv")
    parser.add_argument(
        "--out-request",
        default=None,
        help="per-request CSV (default: request_metrics.csv next to --out-raw; "
        "written only for full-detail service runs)",
    )
    parser.add_argument(
        "--group-config",
        default=None,
        help="JSON comparison group config (see module docstring)",
    )
    parser.add_argument(
        "--normalization",
        choices=[NORMALIZATION_GROUP_MAX, NORMALIZATION_RATIO_TO_BASELINE],
        default=NORMALIZATION_GROUP_MAX,
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="baseline run_id (ratio_to_baseline; overrides group config)",
    )
    args = parser.parse_args(argv)

    log_paths = [Path(log) for log in args.logs]
    for log_path in log_paths:
        # R5-F1：先做 gz 透明回退再判存在（字面 cpp.log 缺失而 .gz 在场
        # 的归档 run_dir 不再 "log not found" 必失败）。
        if not _resolve_log_path(log_path).is_file():
            print(f"postprocess error: log not found: {log_path}", file=sys.stderr)
            return 1
    group_config = None
    if args.group_config is not None:
        try:
            group_config = json.loads(Path(args.group_config).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            print(f"postprocess error: cannot read group config: {error}", file=sys.stderr)
            return 1

    try:
        runs = _parse_logs(log_paths)
        if not runs:
            raise PostprocessError("no [METRIC] runs found in the input logs")
        groups, run_configs = build_groups(runs, group_config)
        raw_rows = build_raw_rows(runs, run_configs)
        normalized_rows = build_normalized_rows(groups, args.normalization, args.baseline)
        request_rows, request_type_counts, request_notes = build_request_metrics(runs)
        # 第一期：die 后端利用率三元组管线断言（在落盘前执行；
        # legacy run 无声明即跳过，行为不变）。
        _check_die_utilization_records(runs)
    except PostprocessError as error:
        print(f"postprocess error: {error}", file=sys.stderr)
        return 1

    _write_csv(Path(args.out_raw), RAW_COLUMNS, raw_rows)
    _write_csv(Path(args.out_normalized), NORMALIZED_COLUMNS, normalized_rows)
    out_request = (
        Path(args.out_request)
        if args.out_request is not None
        else Path(args.out_raw).with_name("request_metrics.csv")
    )
    if request_rows:
        _write_request_csv(out_request, REQUEST_METRICS_COLUMNS, request_rows)
    elif out_request.exists():
        out_request.unlink()
    print(
        json.dumps(
            {
                "runs": len(runs),
                "groups": len(groups),
                "raw_rows": len(raw_rows),
                "normalized_rows": len(normalized_rows),
                "out_raw": args.out_raw,
                "out_normalized": args.out_normalized,
                "request_rows": len(request_rows),
                "request_type_counts": request_type_counts,
                "request_metrics": (
                    str(out_request)
                    if request_rows
                    else "not written: no full-detail request records"
                ),
                "request_metrics_notes": request_notes,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
