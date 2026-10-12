#!/usr/bin/env python3
"""Cross-run metrics post-processing (implementation doc sec.11).

Input: one or more run logs.  Only lines starting with ``[METRIC] `` are
parsed, each holding one single-line JSON record (sec.11.1).  Records are
grouped into runs (one run per ``init`` record occurrence per log).  When a
run's init record carries ``manifest_path``, the referenced
``metrics_manifest.json`` (the C++ collector's per-request manifest) and its
sibling ``manifest.json`` (model/hardware/kv_policy labels) are read as the
run's own declared provenance; all metrics themselves come from the log.
Sidecar resolution (P2, mirrors slo_common.load_request_manifest): a
``metrics_manifest.json`` sitting next to the log (the runner copies it into
the run dir, so archived runs stay self-contained) wins over the init
record's ``manifest_path``, whose generated/ copy may be gone after a
bare-repo restore.

Output (sec.11.2/11.3):

- ``raw_metrics.csv``: one row per measured entity with an explicit ``source``
  column — ``simulator`` (service run latency/tput summary) and
  ``planner_memory_ledger`` (per-rank capacity time-average and peak rows,
  sec.7.8/7.7).
- ``normalized_metrics.csv``: per comparison group, the two frozen methods
  (sec.3.6) with full denominator provenance (``normalization_method``,
  ``normalization_group_id``, ``baseline_run_id``, ``denominator_*``).

- ``request_metrics.csv``: one row per request, frozen column set
  (B1/WP1; later work packages only fill ``NA`` placeholders, never add
  columns).  Built from the ``type=request`` ``[METRIC]`` records (full
  detail level only) joined fail-closed to the run's
  ``metrics_manifest.json``/``manifest.json`` request entries by
  ``queue_index``/``request_id``; ordering/decomposition invariants
  (arrival <= completion, e2e == completion - arrival, stage sum == e2e;
  once the ``first_token`` column is filled -- exact or the WP9
  train-interpolated proxy -- also arrival <= first_token <= completion)
  are re-verified and any violation aborts.  The proxy value additionally
  follows the WP9 clamp/pin rules (clamped/pinned to completion; see
  ``_first_token_proxy_value``).
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
   comparison group;
7. a ``type=consistency`` record reporting ``ok=false``, a non-empty
   ``violations`` list, or ``dropped_events > 0`` (doc sec.12.6: the C++
   collector's fail-closed audit must end with a zero violation count,
   README "违规数必须为 0"); a completed run without any consistency record
   is treated as a truncated/lost audit record and fails as well;
8. a full-detail run whose summary claims completed/incomplete requests but
   carries no ``type=request`` records (the join input was truncated/lost;
   below full detail the C++ emits no request records, so an empty input is
   legal there).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
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
SOURCE_PLANNER_MEMORY = "planner_memory_ledger"

# 2026-09-26 死路径清理（A4-5/M7-3）：microbenchmark 指标路径
# （SOURCE_MICROBENCHMARK / run_mode==microbenchmark 分支 / metrics_manifest
# microbenchmark 点位键）与 planner_roofline 摄入路径（SOURCE_PLANNER_ROOFLINE /
# planner_roofline_stats.json sidecar / kv_bin 等列）在本仓均无生产者
# （plan_materializer 写出的 metrics_manifest.json 只含
# schema_version/repo_variant/manifest_source/requests/slo_sampling；离线
# 规划器入口已删），摄入与输出列一并删除；日志里若出现这两类记录，按未知
# 类型容忍不消费。

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
    "mean_e2e_ns",
    "p50_e2e_ns",
    "p95_e2e_ns",
    "p99_e2e_ns",
    "drain_tput_rps",
    "sim_window_tput_rps",
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

# 2026-09-30 补清（A4-5 收尾）：benchmark_point_id / denominator_iteration_time /
# normalized_iteration_time 三列是 microbenchmark 摄入路径（A4-5 已删）的输出
# 残留——删除后无任何生产者与消费者，且与本文件 A4-5 注释"摄入与输出列
# 一并删除"的声明矛盾，现补齐删除。
NORMALIZED_COLUMNS = [
    "run_id",
    "repo_variant",
    "source",
    "normalization_group_id",
    "normalization_method",
    "baseline_run_id",
    "denominator_time",
    "denominator_tput",
    "normalized_time",
    "normalized_tput",
]


class PostprocessError(RuntimeError):
    """A sec.11.4 failure condition."""


@dataclass
class Run:
    run_id: str
    log_path: Path
    init: dict[str, Any]
    summaries: list[dict[str, Any]] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)
    capacity_timeavg: list[dict[str, Any]] = field(default_factory=list)
    planner_peaks: list[dict[str, Any]] = field(default_factory=list)
    consistency: list[dict[str, Any]] = field(default_factory=list)
    rank_compute: list[dict[str, Any]] = field(default_factory=list)
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
    """R5-F1 (2026-09-29, Python 侧): 官方 runner 归档后 cpp.log 落为
    cpp.log.gz，README ⑤ 步的字面 cpp.log 路径会 "log not found" 必失败。
    对齐同链范式（slo_common / ledger_reconcile 的透明 .gz 回退）：字面
    路径缺失而 <path>.gz 在场时按 gz 解析；两者皆缺时原样返回（由调用方
    统一报 not found）。"""
    if log_path.is_file() or str(log_path).endswith(".gz"):
        return log_path
    gz_path = log_path.with_name(log_path.name + ".gz")
    if gz_path.is_file():
        return gz_path
    return log_path


def _parse_logs(log_paths: Sequence[Path]) -> list[Run]:
    runs: list[Run] = []
    for log_path in log_paths:
        current: Optional[Run] = None
        run_index = 0
        # R5-F1: 归档 gz 形态透明解析(_resolve_log_path 已把字面路径
        # 解析到 <path>.gz;此处按后缀选 opener)。
        opener = (
            gzip.open if str(log_path).endswith(".gz") else open)
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
                elif record_type == "capacity_timeavg":
                    # 注（2026-09-30 登记）：capacity_timeavg / planner_memory_peaks
                    # 两类记录在本仓无生产者（C++ 仅在 metrics manifest 携带
                    # memory_actions/planner_memory_peaks 键时发射，而本仓
                    # plan_materializer 从不写这两个键；写入方在姊妹仓）。
                    # 摄入路径与 RAW_COLUMNS 对应列按跨仓 [METRIC] schema
                    # 可比性保留——本仓正常 run 的这些列恒空。
                    current.capacity_timeavg.append(record)
                elif record_type == "planner_memory_peaks":
                    current.planner_peaks.append(record)
                elif record_type == "consistency":
                    current.consistency.append(record)
                elif record_type == "rank_compute":
                    # 阶段 E2：rank_compute 记录不再 tolerated-unused——
                    # die 后端 run 的利用率三元组（active_kernel_die_*）
                    # 管线断言消费它们（_check_die_utilization_records）；
                    # 其他列照旧不产出 CSV。
                    current.rank_compute.append(record)
                # memory_anchor, iteration (A4-5: the
                # microbenchmark/rank-compute path had no consumer left in
                # this repo), planner_roofline_iteration_stats (M7-3: no
                # producer in this repo) and unknown types are tolerated
                # but unused.
    seen: dict[str, Run] = {}
    deduped: list[Run] = []
    for run in runs:
        previous = seen.get(run.run_id)
        if previous is not None:
            # M7-2: a repeated run_id with an identical summary is the same
            # run seen twice (e.g. the same log passed twice) -- keep one Run
            # so raw/normalized rows do not double.  Conflicting summaries
            # still fail (sec.11.4).
            if previous.summary != run.summary:
                raise PostprocessError(
                    f"run_id {run.run_id!r} appears in multiple logs with "
                    "conflicting summaries (sec.11.4)"
                )
            continue
        seen[run.run_id] = run
        deduped.append(run)
    runs = deduped
    for run in runs:
        if run.schema_versions != {SUPPORTED_SCHEMA}:
            raise PostprocessError(
                f"run {run.run_id}: unsupported or mixed metric schema "
                f"versions {sorted(run.schema_versions)} (sec.11.4); this "
                f"postprocessor supports schema {SUPPORTED_SCHEMA} only"
            )
    return runs


def _load_manifest_sidecars(
    run: Run,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return (metrics_manifest, service manifest.json).

    A4-4: resolution mirrors slo_common.load_request_manifest (runner P2) --
    a ``metrics_manifest.json`` next to the log wins over the init record's
    ``manifest_path``, so an archived run dir (which carries its own copy)
    postprocesses offline even after the generated/ original is gone.  The
    sibling service ``manifest.json`` is resolved next to whichever copy was
    used; it is OPTIONAL (the runner copies it only when the ET dir has one)
    and its absence only degrades the label fallbacks -- that degradation is
    reported on stderr, never silent (A4-3).  A sidecar that exists but
    cannot be read or parsed fails closed; so does a run whose
    ``manifest_path`` is declared but resolves to no readable manifest.
    """

    candidates: list[Path] = []
    local = run.log_path.parent / "metrics_manifest.json"
    if local.is_file():
        candidates.append(local)
    manifest_path = run.init.get("manifest_path")
    if manifest_path:
        candidates.append(Path(manifest_path))
    if not candidates:
        print(
            f"[metrics_postprocess] note: run {run.run_id}: no "
            "metrics_manifest.json sidecar and no init manifest_path "
            "(labels fall back to the group config and init fields)",
            file=sys.stderr,
        )
        return {}, {}
    metrics_path = candidates[0]
    try:
        metrics_manifest = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PostprocessError(
            f"run {run.run_id}: cannot read the run's declared metrics "
            f"manifest {metrics_path}: {error} (fail-closed; no silent "
            "degradation)"
        ) from None
    if not isinstance(metrics_manifest, dict):
        raise PostprocessError(
            f"run {run.run_id}: metrics manifest {metrics_path} must contain "
            "a JSON object (fail-closed)"
        )
    service_path = metrics_path.parent / "manifest.json"
    if service_path.is_file():
        try:
            service_manifest = json.loads(service_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PostprocessError(
                f"run {run.run_id}: cannot read the sibling service manifest "
                f"{service_path}: {error} (fail-closed)"
            ) from None
        if not isinstance(service_manifest, dict):
            raise PostprocessError(
                f"run {run.run_id}: service manifest {service_path} must "
                "contain a JSON object (fail-closed)"
            )
    else:
        # A4-3: an ABSENT optional manifest is reported (stderr) and returned
        # as None so downstream consumers can tell "no file" apart from
        # "file present" -- the request-count gate below only checks manifests
        # that actually exist.
        service_manifest = None
        print(
            f"[metrics_postprocess] note: run {run.run_id}: no service "
            f"manifest.json at {service_path} (optional input; model/"
            "dataset/hardware/kv_policy labels fall back to the group "
            "config and init fields)",
            file=sys.stderr,
        )
    return metrics_manifest, service_manifest


def _check_request_completion(run: Run) -> None:
    """sec.11.4: an incomplete request must never be silently counted."""

    summary = run.summary
    if not run.requests:
        # A4-18: an empty request record set is legal ONLY below full detail
        # (the C++ collector emits type=request records at --metrics-detail
        # full and nothing below).  At full detail the summary claims must be
        # backed by records: claims > 0 with zero records means the log was
        # truncated/lost -> fail closed.  Claims of 0/0 with zero records is
        # a legitimately empty request queue and has nothing to cross-check.
        detail_level = run.init.get("detail_level")
        if detail_level == "full":
            claimed_completed = summary.get("completed_unique_requests") or 0
            claimed_incomplete = summary.get("incomplete_requests") or 0
            if claimed_completed or claimed_incomplete:
                raise PostprocessError(
                    f"run {run.run_id}: full-detail log claims "
                    f"completed_unique_requests={claimed_completed} "
                    f"incomplete_requests={claimed_incomplete} but carries "
                    "no type=request records (log truncated or records "
                    "lost; fail-closed)"
                )
            return
        if detail_level in ("summary", "off"):
            return  # summary/off detail level: nothing to cross-check against
        raise PostprocessError(
            f"run {run.run_id}: init record carries no recognized "
            f"detail_level (got {detail_level!r}); cannot tell whether "
            "request records are expected (fail-closed)"
        )
    completed_records = sum(
        1 for record in run.requests
        if _explicit_bool(record.get("completed"),
                          field="completed", run_id=run.run_id))
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


def _check_consistency_records(runs: Sequence[Run]) -> None:
    """A4-1 hard gate (doc sec.12.6; README "违规数必须为 0").

    The C++ collector's type=consistency record is its fail-closed self
    audit: ok=false (any violation or dropped event) previously reached this
    module and was dropped, letting a violating run postprocess with exit 0.
    Any such record now aborts.  A completed run without any consistency
    record is equally a lost audit trail: the collector emits exactly one
    per run, so absence means truncation (fail-closed).
    """

    for run in runs:
        if not run.consistency:
            raise PostprocessError(
                f"run {run.run_id}: no type=consistency record found (the "
                "collector emits exactly one per run; the audit trail is "
                "truncated, fail-closed)"
            )
        for index, record in enumerate(run.consistency):
            # R9-F2 (2026-09-29): isinstance 作校验器是 fail-open——str
            # "3"/float 2.0 时旧守卫 isinstance(dropped, int) 为假,自审计
            # 静默跳过。类型漂移本身即缺陷:非 int 一律 fail-closed
            # (bool 亦拒;字段缺失沿用旧缺省 0)。
            dropped = record.get("dropped_events")
            if dropped is not None and (
                    isinstance(dropped, bool)
                    or not isinstance(dropped, int)):
                raise PostprocessError(
                    f"run {run.run_id}: consistency record #{index} "
                    f"dropped_events must be int, got {dropped!r} "
                    "(collector self-audit record is corrupt, fail-closed)"
                )
            if dropped is None:
                dropped = 0
            violations = record.get("violations") or []
            if (
                record.get("ok") is not True
                or dropped > 0
                or violations
            ):
                raise PostprocessError(
                    f"run {run.run_id}: consistency record #{index} failed "
                    f"the collector self-audit (ok={record.get('ok')!r}, "
                    f"dropped_events={dropped}, violations={violations}); "
                    "a run with consistency violations must not postprocess "
                    "as success (sec.12.6, README violation-count contract)"
                )


# 阶段 E2：die 后端利用率三元组的管线级断言。C++ MetricCollector 在每条
# type=rank_compute 记录上输出 active_kernel_die_{compute,memory,vector}_util
# （门控=compute-cost-model=die；roofline 键保持 legacy 门控不动）。本断言
# 在后处理侧 fail-closed：声明 die 后端的 run，每个 rank 的三个键必须
# 非 null、数值、且落在 [0, 1+eps]（eps 与 C++ kUtilEpsilon 同值）。
DIE_UTILIZATION_KEYS = (
    "active_kernel_die_compute_util",
    "active_kernel_die_memory_util",
    "active_kernel_die_vector_util",
)
# 与 C++ MetricCollector 的 kUtilEpsilon 一致（上行在 1e-6 容差内钳制）。
DIE_UTILIZATION_EPSILON = 1e-6
# null 的唯一合法理由：窗口内没有任何 comp 节点（C++ 端 note 原文）。
_DIE_NULL_NOTE_OK = "unavailable: no comp nodes in the utilization window"


def _run_declares_die_backend(run: Run) -> bool:
    """该 run 是否声明 die 计算后端。

    声明源优先级：metrics_manifest.json 的 compute-cost-model（阶段 E2
    起由 plan_materializer 同步写入，归档 run 目录自带）> 同目录
    manifest.json 的同名键（阶段 C 的服务 manifest）> 无声明时从
    rank_compute 记录自身推断（C++ 的 note 文案区分 not-die）。
    """

    metrics_manifest, service_manifest = _load_manifest_sidecars(run)
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


def _check_die_utilization_records(runs: Sequence[Run]) -> None:
    """Pipeline assertion (stage E2): die-backend utilization triplet.

    A run that declares the die compute backend must carry, on EVERY
    rank_compute record, the three active_kernel_die_* utilization values as
    non-null numbers within [0, 1+eps]; a null is legal only with the
    collector's "no comp nodes" note.  Anything else (missing keys, a null
    with an unknown note, out-of-range or non-numeric values, or a die run
    with no rank_compute records at all) fails the postprocess fail-closed.
    """

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
                        "rank; only the no-comp-nodes note is a legal null)"
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

REQUEST_TYPE_HUMAN = "human"
REQUEST_TYPE_TOOL = "tool"
REQUEST_TYPE_UNKNOWN = "unknown"
NA = "NA"
# WP9 首 token 来源标记(request_metrics.csv first_token_source;主规格
# §3.1:A 类枚举 exact|train_interpolated|NA。exact = 事件码 8;NA =
# 无事件且归档 train ledger 不可得;train_interpolated = B4 fallback
# proxy,display-only,SLO 判定路径禁入)。
FIRST_TOKEN_SOURCE_EXACT = "exact"
FIRST_TOKEN_SOURCE_TRAIN_INTERPOLATED = "train_interpolated"


def _explicit_bool(value: Any, *, field: str, run_id: str) -> bool:
    """R9-F2 (2026-09-29): 显式布尔解析——仅接受真实 bool 或明确字面量
    ("true"/"false"/"1"/"0",大小写不敏感);None = 字段缺失按 False(与
    改前缺省语义一致)。旧 bool(value) 真值判断对任意非空字符串恒真
    ("false" → True 把 failed 判 completed、"false" baseline 判成
    baseline——fail-open);其余类型一律 PostprocessError fail-closed。"""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "1"):
            return True
        if text in ("false", "0"):
            return False
    raise PostprocessError(
        f"run {run_id}: field {field!r} must be a bool or an explicit "
        f"true/false literal, got {value!r}"
    )


def _strict_int_cell(value: Any) -> Any:
    """R9-F1 (2026-09-29): 指标 CSV 数值列的严格整型口径——仅接受真
    int(bool 拒);None/float/str 等一律渲染 NA,不再裸 int() 静默截断
    (5.7→5)或强转("5"→5)——错型漂移绝不改写数值。"""
    if value is None or isinstance(value, bool) or not isinstance(value, int):
        return NA
    return value


def _int_or_na(value: Any) -> Any:
    """Render a metrics JSON integer, or the frozen NA placeholder.

    (R9-F1, 2026-09-29) 改走 _strict_int_cell 严格口径(旧裸 int():
    float 静默截断/str 接受/bool=1)。"""
    return _strict_int_cell(value)


def _int_or_none(value: Any) -> Optional[int]:
    """Metrics JSON integer or None (WP9 proxy arithmetic helper).

    严格整型口径（与渲染侧 _strict_int_cell 的 R9-F1 纪律一致）：浮点/字符串
    等错型直接抛错，绝不静默截断——first_token_ns 等值一旦从 uint64 漂移成
    float，截断后会以 exact 身份写进 CSV。None（缺值）与 bool 照旧归 None。
    登记的口径梯度不对称（2026-09-30）：整册台账损坏走 R9-A2 的"降级 NA
    继续"，而这里单行错型是硬失败——display-only proxy 的单行脏值比整册
    损坏更严厉，属有意选择（错型静默降级会掩盖协议漂移），如需放宽应连同
    R9-A2 一起重裁。
    """
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, int):
        raise PostprocessError(
            f"expected an integer metric value, got {type(value).__name__}: "
            f"{value!r}")
    return value


def _manifest_request_entries(
    run: Run,
    metrics_manifest: dict[str, Any],
    service_manifest: dict[str, Any] | None,
) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    """Per-request manifest entries keyed by queue_index (fail-closed join).

    Merges the run's ``metrics_manifest.json`` (the C++ collector input)
    with its sibling service ``manifest.json``.  Merge priority (A4-17):
    manifests are processed in the order metrics_manifest.json then
    manifest.json, and on a duplicate ``queue_index`` the LATER manifest's
    (service manifest.json) values silently override the earlier one's for
    any same-named field; the earlier manifest only fills fields the later
    one lacks.  The one field whose conflict IS checked is ``request_id``:
    both manifests must agree on it for the same queue_index or the run
    aborts.  Returns ``(entries_by_queue_index,
    per_manifest_request_counts)``.
    """
    merged: dict[int, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for name, manifest in (
        ("metrics_manifest.json", metrics_manifest),
        ("manifest.json", service_manifest),
    ):
        if manifest is None:
            # Absent OPTIONAL manifest (A4-3): nothing was declared, so the
            # declared-vs-carried count gate has nothing to check -- skipping
            # it here is what keeps the documented "absence only degrades
            # labels" contract true instead of aborting the whole run.
            continue
        requests = manifest.get("requests")
        if not isinstance(requests, list):
            counts[name] = 0
            continue
        counts[name] = len(requests)
        for entry in requests:
            if not isinstance(entry, dict) or "queue_index" not in entry:
                raise PostprocessError(
                    f"run {run.run_id}: {name} request entry without "
                    "queue_index cannot be joined to the request records"
                )
            queue_index = entry["queue_index"]
            # (2026-10-01)与记录侧 R9-F1 同款严格 int 口径:不可哈希
            # queue_index 曾在下方 dict 键操作处以裸 TypeError 逃出
            # PostprocessError 通道;float/bool 等可哈希错型曾借相等性
            # (1.0==1、True==1)静默误联到 int 键条目——错型绝不静默折算。
            if isinstance(queue_index, bool) or not isinstance(queue_index, int):
                raise PostprocessError(
                    f"run {run.run_id}: {name} request entry queue_index "
                    f"must be int, got {queue_index!r} (fail-closed)"
                )
            if queue_index in merged:
                previous = merged[queue_index]
                if previous.get("request_id") != entry.get("request_id"):
                    raise PostprocessError(
                        f"run {run.run_id}: queue_index {queue_index} has "
                        f"conflicting request_id across manifests "
                        f"({previous.get('request_id')!r} vs "
                        f"{entry.get('request_id')!r})"
                    )
                combined = dict(entry)
                for key, value in previous.items():
                    combined.setdefault(key, value)
                merged[queue_index] = combined
            else:
                merged[queue_index] = dict(entry)
    if not merged:
        raise PostprocessError(
            f"run {run.run_id}: no request entries found in either "
            "metrics_manifest.json or manifest.json; the request records "
            "cannot be joined (fail-closed)"
        )
    return merged, counts


def _check_request_row_invariants(
    run_id: str, record: dict[str, Any], queue_index: Any
) -> None:
    """Ordering/decomposition invariants on rows that carry values.

    Checked only among the fields that are present: arrival <= first_token
    <= completion (first_token skipped while NA), e2e == completion -
    arrival, and queue+prefill+gap+decode == e2e.  Any violation aborts
    (no silent clamping).
    """
    arrival = record.get("arrival_ns")
    first_token = record.get("first_token_ns")
    completion = record.get("completion_ns")
    e2e = record.get("e2e_ns")
    if arrival is None or completion is None:
        return
    if arrival > completion:
        raise PostprocessError(
            f"run {run_id}: queue_index {queue_index}: arrival_ns "
            f"{arrival} > completion_ns {completion}"
        )
    if first_token is not None and not (arrival <= first_token <= completion):
        raise PostprocessError(
            f"run {run_id}: queue_index {queue_index}: first_token_ns "
            f"{first_token} outside [arrival_ns {arrival}, "
            f"completion_ns {completion}]"
        )
    if e2e is not None and e2e != completion - arrival:
        raise PostprocessError(
            f"run {run_id}: queue_index {queue_index}: e2e_ns {e2e} != "
            f"completion_ns - arrival_ns ({completion} - {arrival} = "
            f"{completion - arrival})"
        )
    parts = [
        record.get("queue_ns"),
        record.get("prefill_ns"),
        record.get("prefill_decode_gap_ns"),
        record.get("decode_ns"),
    ]
    if e2e is not None and all(part is not None for part in parts):
        if sum(parts) != e2e:
            raise PostprocessError(
                f"run {run_id}: queue_index {queue_index}: stage breakdown "
                f"{parts} sums to {sum(parts)} != e2e_ns {e2e}"
            )


# ---------------------------------------------------------------------------
# WP9 fallback (B4, 2026-08-26): train-interpolated first-token proxy.
#
# B3_FACE 60s gate-2 (decision equivalence) failed -- the split's physical
# perturbation is amplified by the closed loop (see
# /tmp/slo_wps/gates/B3_FACE.DONE, evidence /tmp/slo_wps/b3/FACE/) -- so the
# default first-token source falls back to the spec 1.6 proxy
# (SH_FIRST_TOKEN_SPLIT now defaults to "0"):
#
#   first_token_proxy = decode_start_ns
#                     + (w1 / sum_i w_i) * (first_train_end_ns - decode_start_ns)
#   w_i = W_bytes + KV_bytes(context + consumed + i)     [i = 1..N, 1-based]
#
# with the weight sum running over the train's iterations (i = 1..N,
# N = the ledger row's train length; S3 cross-repo ruling).  The w_i mirror
# the scheduler's own pass-span encoding for a train member: span step i is
# (1, context + consumed + i).  The debut's first token completes with the
# train's FIRST iteration, so the interpolation weight is iteration 1's
# weight over the train's N iteration weights.  The debut joins decode
# exactly once with consumed=0 (sticky decode; verified 1454/1454 joiner
# entries == requests, each request joining exactly once, in the 60s
# reference run), hence consumed=0.
#
# first_train_end_ns: the train ledger row's tick is the emission (start)
# boundary; a face instance is busy until the train's end barrier completes
# (busy gate = one train in flight, face_online_scheduler.py), so the NEXT
# ledger row on the same instance is the first decision boundary after the
# end barrier and is used as the train-end tick (last train of an instance
# -> proxy not obtainable -> NA).  face mixed trains (prefill chunks of the
# queue head interleaved with decode iterations) share the same busy gate;
# the boundary therefore also covers the mixed-in chunk work of the train.
# Degenerate 1-iteration trains (N=1 -> share=1) make the proxy equal that
# next-emission boundary, which sits after the barrier tick recorded as
# completion; such values are clamped to completion (the first token
# physically cannot complete after the request) and flagged
# "clamped_to_completion" in instructions.  Symmetrically, a
# decode_length==1 debut riding a MULTI-iteration train gets share<1 and
# can land BEFORE the recorded completion tick; exact-mode semantics for
# decode_length==1 is first_token==completion, so those are pinned to
# completion and flagged "pinned_to_completion" (2026-09-05 fix: face 0902
# full_rt had 18/268 dl1 rows up to 4.9s early).
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
        # utf-8-sig matches the authoritative loader
        # (generate_face_trace.load_face_trace_config, which also strips the
        # row cells): a BOM would otherwise turn the first field name into
        # "\ufeffkind" and silently drop every config row (proxy degrades to
        # NA instead of erroring) -- an avoidable lexicon split (2026-10-01).
        with config_path.open(encoding="utf-8-sig") as source:
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
        # R9-A2 (2026-09-29): ledger 在场却不可读时的丢弃原因("文件名
        # (异常类型)");None = 单纯缺失。下游降级 note 据此区分"缺失"
        # 与"损坏丢弃"——旧写法两者同词 no_train_ledger,审计员被误导为
        # 未归档。
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
            # R9-A2 (2026-09-29): 在场却损坏 ≠ 缺失——记丢弃原因供降级
            # note 区分(整册丢弃的降级语义本身保留:proxy 是 display-only
            # fallback,损坏不阻断后处理)。
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
    record: dict[str, Any],
    entry: dict[str, Any],
) -> tuple[Optional[int], str]:
    """(proxy_ns, note) for one request; (None, reason-note) when the proxy
    is not obtainable from the archived artifacts."""

    request_id = str(record.get("request_id") or "")
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
        # (2026-10-01) tier-2 only applies when the history field is
        # ACTUALLY present: `or 0` treated a missing field as 0 and silently
        # under-computed the context (a guess, against the module's
        # never-a-guess rule); a missing history now falls through to the
        # exact tier-3 derivation (or NA) instead.
        history = _int_or_none(entry.get("history_tokens_before"))
        prefill = _int_or_none(entry.get("prefill_length"))
        if prefill is not None and history is not None:
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
    # and overshoot completion (S3's 60s reference run: 7/1454 rows).
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
    # end-barrier tick recorded as completion (delivery+scheduling gap).
    # Clamp to completion and say so -- for decode_length==1 the exact-mode
    # semantics is first_token==completion anyway.
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


def _request_metric_rows(
    run: Run, entries: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    """request_metrics.csv rows (frozen column set; B1/WP1 fill level).

    Timing/stage columns come from the C++ ``type=request`` records; length
    columns come from the joined manifest entries (WP2 passthrough) and
    degrade to ``NA`` with a per-row ``instructions`` note -- never a
    guess.  ``request_type`` degrades to ``unknown`` with a note only when
    the manifest field is absent; a present but unexpected passthrough
    value (anything other than human/tool) is replaced by ``unknown``
    without a note.
    """
    rows: list[dict[str, Any]] = []
    # WP9 fallback (B4): train-interpolated proxy index over the run's
    # train_ledger.jsonl (loaded lazily; absent ledger -> NA with a note).
    proxy_index = _TrainProxyIndex(run)
    proxy_available = proxy_index.available()
    for record in run.requests:
        queue_index = record.get("queue_index")
        # R9-F1 (2026-09-29): 行身份列严格 int——bool/float/str 曾被裸
        # int() 静默折算(queue_index=bool 曾借 dict 键 True==1 误联到
        # queue 1 的 manifest 条目)。(2026-10-01)校验上提到 join 之前:
        # 不可哈希 queue_index(list/dict 等)曾先在下方 entries.get 处以
        # 裸 TypeError 逃出 PostprocessError 通道;上提后损坏记录一律先走
        # 本 fail-closed 报错,再进 join。
        if isinstance(queue_index, bool) or not isinstance(queue_index, int):
            raise PostprocessError(
                f"run {run.run_id}: request record queue_index must be "
                f"int, got {queue_index!r} (fail-closed)"
            )
        entry = entries.get(queue_index)
        if entry is None:
            raise PostprocessError(
                f"run {run.run_id}: request record queue_index="
                f"{queue_index!r} cannot be joined to any manifest request "
                "entry (fail-closed; rows are never skipped or truncated)"
            )
        manifest_request_id = entry.get("request_id")
        if manifest_request_id not in (None, record.get("request_id")):
            raise PostprocessError(
                f"run {run.run_id}: queue_index {queue_index} "
                f"request_id mismatch: record {record.get('request_id')!r} "
                f"vs manifest {manifest_request_id!r}"
            )
        notes: list[str] = []
        # R9-F2 (2026-09-29): 显式布尔解析——bool("false") 恒真曾把
        # 字符串化的 failed 判成 completed。
        completed = _explicit_bool(record.get("completed"),
                                   field="completed", run_id=run.run_id)
        # A4-9（2026-09-26）：request 级 terminal_status 权威词表就是
        # {"completed", "failed"}（本处派生自 C++ completed 布尔；与
        # slo_common.TERMINAL_STATUSES 及契约测试 TERMINAL_STATUS_VALUES
        # 单一口径对齐；failed 覆盖全部未完成情形）。
        terminal_status = "completed" if completed else "failed"
        if not completed:
            reason = record.get("incomplete_reason") or "unspecified"
            notes.append(f"incomplete_reason={reason}")
        # WP9 首 token（2026-08-26）：exact code-8 值优先；缺失且归档
        # train ledger 可得时填 train-interpolated proxy（B4 fallback，
        # display-only，SLO 判定路径禁入）；否则 NA（instructions 记原因）。
        first_token_ns = _int_or_none(record.get("first_token_ns"))
        first_token_source = (
            FIRST_TOKEN_SOURCE_EXACT if first_token_ns is not None else None)
        if first_token_ns is None and proxy_available:
            proxy_ns, proxy_note = _first_token_proxy_value(
                proxy_index, record, entry)
            if proxy_ns is not None:
                first_token_ns = proxy_ns
                first_token_source = FIRST_TOKEN_SOURCE_TRAIN_INTERPOLATED
            notes.append(proxy_note)
        elif first_token_ns is None:
            if proxy_index.load_failure:
                # R9-A2 (2026-09-29): ledger 在场却损坏被丢弃——note 明说
                # 损坏,不再冒充 no_train_ledger(缺失)。
                notes.append(
                    "proxy_unavailable:train_ledger_corrupt({})".format(
                        proxy_index.load_failure))
            else:
                notes.append("proxy_unavailable:no_train_ledger(results/)")
        if first_token_ns is not None:
            # Ordering invariant is re-checked fail-closed on the FILLED
            # value (exact or proxy): arrival <= first_token <= completion.
            checked = dict(record)
            checked["first_token_ns"] = first_token_ns
            _check_request_row_invariants(run.run_id, checked, queue_index)
        else:
            _check_request_row_invariants(run.run_id, record, queue_index)
        request_type = entry.get("request_type")
        if request_type not in (REQUEST_TYPE_HUMAN, REQUEST_TYPE_TOOL):
            if request_type is None:
                notes.append("request_type=unknown(manifest field absent)")
            request_type = REQUEST_TYPE_UNKNOWN
        lengths: dict[str, Any] = {}
        for column in ("prefill_length", "decode_length", "prefix_len"):
            value = entry.get(column)
            if value is None:
                notes.append(f"{column}={NA}(manifest field absent)")
                lengths[column] = NA
            elif isinstance(value, bool) or not isinstance(value, int):
                # R9-F1 (2026-09-29): 裸 int() 对 float 截断/str 强转——
                # 错型一律 NA + 注记,绝不静默改写数值。
                notes.append(f"{column}={NA}(manifest field not an int)")
                lengths[column] = NA
            else:
                lengths[column] = value
        # R9-F1 (2026-09-29): turn_index 同款严格 int 口径(queue_index 的
        # 校验已上提到本函数 join 之前,见上方注记)。
        turn_index = record.get("turn_index", 0)
        if isinstance(turn_index, bool) or not isinstance(turn_index, int):
            raise PostprocessError(
                f"run {run.run_id}: request record turn_index must be "
                f"int, got {turn_index!r} (queue_index={queue_index}, "
                "fail-closed)"
            )
        rows.append(
            {
                "queue_index": queue_index,
                "request_id": record.get("request_id", ""),
                "session_id": record.get("session_id", ""),
                "turn_index": turn_index,
                "request_type": request_type,
                "terminal_status": terminal_status,
                "arrival_ns": _int_or_na(record.get("arrival_ns")),
                "prefill_start_ns": _int_or_na(record.get("prefill_start_ns")),
                "prefill_end_ns": _int_or_na(record.get("prefill_end_ns")),
                "decode_start_ns": _int_or_na(record.get("decode_start_ns")),
                # WP9 首 token(2026-08-26):exact = request 记录的
                # first_token_ns(事件码 8,split ON 时由首步批 first_token
                # 标记产出);缺失时 train-interpolated proxy(B4 fallback,
                # display-only;见 _TrainProxyIndex 块);均无 → NA。
                "first_token_ns": (
                    NA if first_token_ns is None else first_token_ns),
                "first_token_source": (
                    first_token_source if first_token_source is not None
                    else NA),
                "completion_ns": _int_or_na(record.get("completion_ns")),
                "queue_ns": _int_or_na(record.get("queue_ns")),
                "prefill_ns": _int_or_na(record.get("prefill_ns")),
                "prefill_decode_gap_ns": _int_or_na(
                    record.get("prefill_decode_gap_ns")
                ),
                "decode_ns": _int_or_na(record.get("decode_ns")),
                "e2e_ns": _int_or_na(record.get("e2e_ns")),
                # WP4/WP5 fill the KV-hit and restore columns; frozen NA here.
                "kv_hit_state": NA,
                "restore_start_ns": NA,
                "restore_complete_ns": NA,
                "pre_prefill_restore_ns": NA,
                "hidden_restore_ns": NA,
                "exposed_restore_stall_ns": NA,
                "hidden_ratio": NA,
                "prefill_length": lengths["prefill_length"],
                "decode_length": lengths["decode_length"],
                "prefix_len": lengths["prefix_len"],
                "instructions": "; ".join(notes),
            }
        )
    return rows


def write_request_metrics(
    runs: Sequence[Run], path: Path
) -> dict[str, Any]:
    """Write the frozen-schema request_metrics.csv next to the raw CSV.

    Full detail only: summary/off logs carry no ``type=request`` records,
    so no file is written and the skip reason is printed (it lands in the
    run's postprocess.log).  Any join failure aborts with a non-zero exit.
    """
    with_requests = [run for run in runs if run.requests]
    if not with_requests:
        detail_levels = sorted(str(run.init.get("detail_level")) for run in runs)
        summary = {
            "out_requests": str(path),
            "request_metrics": "skipped",
            "reason": (
                "no type=request [METRIC] records in the input logs "
                "(request rows are emitted at --metrics-detail=full only)"
            ),
            "detail_levels": detail_levels,
        }
        print(json.dumps(summary))
        return summary
    if len(with_requests) > 1:
        raise PostprocessError(
            "request_metrics.csv is a single-run product; "
            f"{len(with_requests)} runs carry request records "
            f"({', '.join(run.run_id for run in with_requests)}); "
            "postprocess one run log at a time"
        )
    run = with_requests[0]
    # 2026-09-26 review: the guard is symmetric with _load_manifest_sidecars
    # now -- a local metrics_manifest.json next to the log satisfies the
    # join requirement on its own (that function already prefers it over
    # the init record's manifest_path); the manifest is unavailable only
    # when BOTH sources are absent.
    has_manifest_path = bool(run.init.get("manifest_path"))
    has_local_sidecar = (
        run.log_path.parent / "metrics_manifest.json"
    ).is_file()
    if not has_manifest_path and not has_local_sidecar:
        raise PostprocessError(
            f"run {run.run_id}: request records cannot be joined "
            "without the init record manifest_path or a "
            "metrics_manifest.json sidecar next to the log (fail-closed)"
        )
    metrics_manifest, service_manifest = _load_manifest_sidecars(run)
    entries, counts = _manifest_request_entries(
        run, metrics_manifest, service_manifest
    )
    for name, count in counts.items():
        if count != len(run.requests):
            raise PostprocessError(
                f"run {run.run_id}: {name} declares {count} requests but "
                f"the log carries {len(run.requests)} request records "
                "(fail-closed; no silent truncation)"
            )
    rows = _request_metric_rows(run, entries)
    _write_csv(path, REQUEST_METRICS_COLUMNS, rows)
    status_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for row in rows:
        status_counts[str(row["terminal_status"])] = (
            status_counts.get(str(row["terminal_status"]), 0) + 1
        )
        type_counts[str(row["request_type"])] = (
            type_counts.get(str(row["request_type"]), 0) + 1
        )
        source_counts[str(row["first_token_source"])] = (
            source_counts.get(str(row["first_token_source"]), 0) + 1
        )
    summary = {
        "out_requests": str(path),
        "request_rows": len(rows),
        "terminal_status_counts": status_counts,
        "request_type_counts": type_counts,
        "first_token_source_counts": source_counts,
        "rows_with_notes": sum(1 for row in rows if row["instructions"]),
    }
    print(json.dumps(summary))
    return summary


def _labels(run: Run, service_manifest: dict[str, Any], run_config: dict[str, Any]) -> dict[str, str]:
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
            "mean_e2e_ns": summary.get("mean_e2e_ns"),
            "p50_e2e_ns": summary.get("p50_e2e_ns"),
            "p95_e2e_ns": summary.get("p95_e2e_ns"),
            "p99_e2e_ns": summary.get("p99_e2e_ns"),
            "drain_tput_rps": summary.get("drain_tput_rps"),
            "sim_window_tput_rps": summary.get("sim_window_tput_rps"),
            "memory_actions_total": summary.get("memory_actions_total", ""),
            "memory_actions_replayed": summary.get("memory_actions_replayed", ""),
            "memory_actions_unresolved": summary.get("memory_actions_unresolved", ""),
        }
    )
    return [row]


def _peak_rank_sort_key(item: tuple[Any, dict[str, Any]]) -> tuple[int, Any, str]:
    """A4-15: order peak rows by integer rank, tolerating a missing (None)
    or malformed rank without the int/None mixed-type TypeError that the
    bare ``sorted(...)`` hit; non-integer ranks sort after integer ones."""

    rank = item[0]
    if isinstance(rank, bool) or not isinstance(rank, int):
        return (1, 0, str(rank))
    return (0, rank, "")


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
        # Peaks without replay still surface.
        for rank, peak in sorted(peaks_by_rank.items(), key=_peak_rank_sort_key):
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


def build_raw_rows(
    runs: Sequence[Run], run_configs: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        _check_request_completion(run)
        _, service_manifest = _load_manifest_sidecars(run)
        labels = _labels(
            run, service_manifest or {}, run_configs.get(run.run_id, {})
        )
        rows.extend(_service_rows(run, labels))
        rows.extend(_memory_ledger_rows(run, labels))
    return rows


@dataclass
class Group:
    # A4-7/M7-4: the former allow_digest_mismatch / labels_by_run fields are
    # deleted -- both were write-only (digest enforcement happens inline in
    # build_groups from the group CONFIG entry; per-run labels flow through
    # the returned run_configs dict), never read downstream.
    group_id: str
    runs: list[Run]
    baseline_run_id: Optional[str]


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
                    baseline_run_id=None,
                )
            )
        return groups, run_configs

    configured_groups = group_config.get("groups")
    if not isinstance(configured_groups, list) or not configured_groups:
        raise PostprocessError("group config must contain a non-empty 'groups' list")
    for entry in configured_groups:
        group_id = entry.get("group_id")
        if not group_id:
            raise PostprocessError("group config entry without group_id")
        members: list[Run] = []
        baseline_run_id = entry.get("baseline_run_id")
        baseline_marks = 0
        for run_entry in entry.get("runs", []):
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
            run_configs[run_id] = labels
            # R9-F2 (2026-09-29): baseline 标记显式布尔解析(字符串
            # "false" 曾被真值判断判成 baseline)。
            if _explicit_bool(
                    run_entry.get("baseline"),
                    field=f"group {group_id} run {run_id} baseline",
                    run_id=run_id):
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
                baseline_run_id=baseline_run_id,
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

        # Service rows: one per run.  (A4-5: the microbenchmark branch is
        # deleted -- run_mode=microbenchmark has no producer in this repo.)
        # B8: a null mean_e2e_ns/drain_tput_rps is the collector's legal
        # zero-completion / zero-length-drain-window summary (MetricCollector
        # emits nullptr there, with drain_tput_note); such a metric is left
        # out of the denominators and normalizes to an empty cell instead of
        # failing the whole postprocess.  A summary record missing the fields
        # outright stays a sec.11.4 failure (malformed collector output).
        service: list[tuple[Run, Optional[float], Optional[float]]] = []
        for run in group.runs:
            summary = run.summary
            if "mean_e2e_ns" not in summary or "drain_tput_rps" not in summary:
                raise PostprocessError(
                    f"run {run.run_id}: summary lacks mean_e2e_ns/"
                    "drain_tput_rps; cannot normalize (sec.11.4)"
                )
            mean_e2e = summary.get("mean_e2e_ns")
            drain_tput = summary.get("drain_tput_rps")
            service.append(
                (
                    run,
                    None if mean_e2e is None else float(mean_e2e),
                    None if drain_tput is None else float(drain_tput),
                )
            )

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
            [mean for _, mean, _ in service if mean is not None],
            None if baseline_service is None else baseline_service[0],
        ) if any(mean is not None for _, mean, _ in service) else None
        tput_den = denominator(
            [tput for _, _, tput in service if tput is not None],
            None if baseline_service is None else baseline_service[1],
        ) if any(tput is not None for _, _, tput in service) else None

        for run, mean_e2e, drain_tput in service:
            time_record = (
                _normalize_value(
                    method=method,
                    metric_name="mean_e2e_ns",
                    raw_value=mean_e2e,
                    group_id=group.group_id,
                    group_values=[mean for _, mean, _ in service if mean is not None],
                    baseline_run_id=baseline_run_id,
                    baseline_value=(
                        None if baseline_service is None else baseline_service[0]
                    ),
                )
                if mean_e2e is not None
                else None
            )
            tput_record = (
                _normalize_value(
                    method=method,
                    metric_name="drain_tput_rps",
                    raw_value=drain_tput,
                    group_id=group.group_id,
                    group_values=[tput for _, _, tput in service if tput is not None],
                    baseline_run_id=baseline_run_id,
                    baseline_value=(
                        None if baseline_service is None else baseline_service[1]
                    ),
                )
                if drain_tput is not None
                else None
            )
            rows.append(
                {
                    "run_id": run.run_id,
                    "repo_variant": run.init.get("repo_variant", ""),
                    "source": SOURCE_SIMULATOR,
                    "normalization_group_id": group.group_id,
                    "normalization_method": method,
                    "baseline_run_id": baseline_run_id or "",
                    "denominator_time": time_den,
                    "denominator_tput": tput_den,
                    "normalized_time": (
                        "" if time_record is None else time_record.normalized_value
                    ),
                    "normalized_tput": (
                        "" if tput_record is None else tput_record.normalized_value
                    ),
                }
            )
    return rows


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    # B8: atomic write -- the rows land in a same-directory temp file
    # (pid-suffixed, so the name can never collide with a real product) that
    # is os.replace()d into place; on any failure the temp file is removed,
    # so a disk-full/quota I/O error cannot leave a truncated or
    # mixed-generation product behind.
    temp_path = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        with temp_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(columns))
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in columns})
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cross-run metrics post-processing (doc sec.11)."
    )
    parser.add_argument("logs", nargs="+", help="run log(s) to parse")
    parser.add_argument("--out-raw", default="raw_metrics.csv")
    parser.add_argument("--out-normalized", default="normalized_metrics.csv")
    parser.add_argument("--out-requests", default="request_metrics.csv")
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

    # R5-F1(2026-09-29):字面 cpp.log 缺失时透明回退 cpp.log.gz(归档
    # 形态;见 _resolve_log_path)。
    log_paths = [_resolve_log_path(Path(log)) for log in args.logs]
    for log_path in log_paths:
        if not log_path.is_file():
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
        _check_consistency_records(runs)
        _check_die_utilization_records(runs)
    except PostprocessError as error:
        print(f"postprocess error: {error}", file=sys.stderr)
        return 1

    # A4-14: every gate runs BEFORE any CSV lands on disk, so a failed gate
    # leaves no half-written raw/normalized products behind.  (B8: all three
    # writes share one error path, and _write_csv lands each file atomically
    # via a same-directory temp + os.replace, so an I/O failure mid-write
    # cannot leave a truncated or mixed-generation product behind.)
    try:
        write_request_metrics(runs, Path(args.out_requests))
        _write_csv(Path(args.out_raw), RAW_COLUMNS, raw_rows)
        _write_csv(Path(args.out_normalized), NORMALIZED_COLUMNS, normalized_rows)
    except PostprocessError as error:
        print(f"postprocess error: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(
            f"postprocess error: cannot write CSV products: {error}",
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "runs": len(runs),
                "groups": len(groups),
                "raw_rows": len(raw_rows),
                "normalized_rows": len(normalized_rows),
                "out_raw": args.out_raw,
                "out_normalized": args.out_normalized,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
