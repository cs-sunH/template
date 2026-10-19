#!/usr/bin/env python3
"""Cross-run metrics post-processing (implementation doc sec.11).

Input: one or more run logs.  Only lines starting with ``[METRIC] `` are
parsed, each holding one single-line JSON record (sec.11.1).  Records are
grouped into runs (one run per ``init`` record occurrence per log).  When a
run's init record carries ``manifest_path``, the referenced
``metrics_manifest.json`` (microbenchmark point spec) and its sibling
``manifest.json`` (model/hardware/kv_policy labels) are read as the run's
own declared provenance; all metrics themselves come from the log.

Output (sec.11.2/11.3):

- ``raw_metrics.csv``: one row per measured entity with an explicit ``source``
  column — ``simulator`` (service run latency/tput summary),
  ``simulator_microbenchmark`` (per benchmark-point iteration rows), and
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
  (arrival <= completion, e2e == completion - arrival, stage sum == e2e)
  are re-verified and any violation aborts.
Failure conditions (sec.11.4, all abort with a non-zero exit):

1. duplicate runs: the same ``run_id`` appearing more than once across the
   input logs (identical summaries included — duplicates would silently
   double-count every downstream row), or two conflicting summary records
   inside one run;
2. explicitly configured comparison group whose workload digests
   (trace_digest / request_mapping_digest) differ without
   ``"allow_digest_mismatch": true``;
3. ``ratio_to_baseline`` selected without exactly one baseline run;
4. a zero normalization denominator;
5. a summary whose completed/incomplete counts are not backed by the run's
   request records (an incomplete request silently counted as complete);
   likewise a run whose init declares ``detail_level=full`` but carries no
   ``type=request`` records at all;
6. mixed metric definition (schema) versions inside one run or one
   comparison group;
7. a run whose collector consistency verdict is not clean (sec.12.6 gate:
   the ``type=consistency`` record's ``ok``/``violations``/``dropped_events``)
   or that carries no ``type=consistency`` record at all — missing
   evidence is a failure, never a pass;
8. a ``[METRIC]`` payload that is not a finite-valued JSON object
   (NaN/Infinity constants and scalar/array top level are rejected at the
   parse layer);
9. duplicate ``queue_index`` entries inside one manifest sidecar of a run
   whose request records must be joined;
10. a ``queue_index`` repeated among the log's own request records, or a
    request-record/manifest ``queue_index`` key-set mismatch — the join
    must be exactly bijective ("rows are never skipped or truncated").
11. a run whose init record declares ``manifest_path`` but neither
    sidecar can be read from the candidate chain (run_dir local copy
    first, then the init manifest_path; missing/moved/unreadable) — the
    previous
    OSError->empty-dict degradation silently emptied the grouping labels
    and collapsed different workloads into one normalization group
    (M-37); the workload digest gate records an explicit SKIP note when
    no run in a group declares any digest, instead of passing vacuously.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from metrics_schema import (  # noqa: E402
    check_schema_version,
    FIRST_TOKEN_SOURCE_EXACT,
    FIRST_TOKEN_SOURCE_TRAIN_INTERPOLATED,
    NA,
    NORMALIZATION_GROUP_MAX,
    NORMALIZATION_RATIO_TO_BASELINE,
    TERMINAL_STATUS_COMPLETED,
    TERMINAL_STATUS_FAILED,
    MetricsSchemaError,
    build_normalization_record,
)

METRIC_PREFIX = "[METRIC] "
SUPPORTED_SCHEMA = 1


def _schema_versions_exact(schemas: set) -> bool:
    """Type-strict ``schemas == {SUPPORTED_SCHEMA}`` equality.

    （2026-10-01 对齐 F2_PY）Python 集合相等吞掉 bool/float 冒充
    （``{True} == {1}``、``{1.0} == {1}``），说谎的生产者用
    ``"schema": true/1.0`` 能同时穿过 run 级与 group 级两道门——
    schema version 必须是字面整数。值经 metrics_schema 自己的检查器
    （单一权威：_require_int + SCHEMA_VERSION 相等）路由，失败降级为
    False -> PostprocessError（带本文件的 run 上下文）。
    """
    if len(schemas) != 1:
        return False
    try:
        check_schema_version(next(iter(schemas)))
    except MetricsSchemaError:
        return False
    return True

SOURCE_SIMULATOR = "simulator"
SOURCE_MICROBENCHMARK = "simulator_microbenchmark"
SOURCE_PLANNER_MEMORY = "planner_memory_ledger"

# kv_policy label 取值集（A.5/2026-09-05：legacy 与 relevant 两个历史
# 变体已清除；session_lru_tiered = session 级二态冷热管理：完整本地/
# 完整远端、整体 LRU 逐出 + 远端池全量恢复；2026-09-25 session 级
# Tiered-LRU 批起原 PARTIAL 半层化三态与两段式逐出已物理移除；收尾后
# 唯一合法值 session_lru_tiered，旧别名 session_lru_recompute 已清除）。
# 与 generate_wsc_llm_trace._parse_config_value 的 kv_cache_policy 值域同源
# （label 与配置键同名同域）；空串 = 未标注（既有产物/合成 manifest 无
# kv_management 节时缺省），保持向后兼容。非空且不在集合内 → fail-closed
# （防止拼错的 label 静默分裂对比组，sec.11.4 同族防线）。
KV_POLICY_LABELS = frozenset({"session_lru_tiered"})

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
    "benchmark_point_id",
    "phase",
    "tp_degree",
    "prefill_chunk",
    "decode_batch",
    "kv_length",
    "repeat_index",
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


def _parse_json_object(payload: str, context: str) -> dict[str, Any]:
    """Parse one JSON document that must be a finite-valued object.

    Parse-layer fail-closed gate (POST-03/POST-04, 2026-09-26): Python's
    ``json`` accepts ``NaN``/``Infinity`` literals by default and any JSON
    top level, so a poisoned metric line used to flow into the products as
    float NaN or die later with a bare ``AttributeError`` on ``.get()``.
    Both are rejected here as structured sec.11.4 failures instead.
    P2 (2026-09-29): ``json`` also parses overflow numbers like ``1e999``
    to ``inf`` without touching ``parse_constant`` — every number in the
    parsed object is therefore re-checked for finiteness recursively.
    """

    def _reject_constant(name: str) -> Any:
        raise ValueError(f"non-finite JSON constant {name}")

    def _reject_non_finite(value: Any) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, float):
            if not math.isfinite(value):
                raise PostprocessError(
                    f"{context}: non-finite JSON number {value!r} "
                    "(fail-closed; sec.11.4)"
                )
        elif isinstance(value, dict):
            for member in value.values():
                _reject_non_finite(member)
        elif isinstance(value, list):
            for member in value:
                _reject_non_finite(member)

    try:
        record = json.loads(payload, parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise PostprocessError(f"{context}: malformed JSON: {error}") from None
    except ValueError as error:
        raise PostprocessError(f"{context}: {error}") from None
    if not isinstance(record, dict):
        raise PostprocessError(
            f"{context}: JSON payload must be an object, got "
            f"{type(record).__name__}"
        )
    _reject_non_finite(record)
    return record


def _resolve_log_path(log_path: Path) -> Path:
    """sync-py-E4 (2026-09-30, ported from the face repo): 官方 runner 归档
    (archive_run_outputs.sh) 把 cpp.log 压缩为 cpp.log.gz 并删除原文件，
    README ⑤ 步的字面 cpp.log 路径会 "log not found" 必失败。对齐同链
    范式（slo_common / ledger_reconcile 的透明 .gz 回退）：字面路径缺失
    而 <path>.gz 在场时按 gz 解析；两者皆缺时原样返回（由调用方统一报
    not found）。"""
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
        # sync-py-E4: 归档 gz 形态透明解析（_resolve_log_path 已把字面
        # 路径解析到 <path>.gz；此处按后缀选 opener）。
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
                record = _parse_json_object(
                    payload, f"{log_path}: [METRIC] JSON line"
                )
                schema = record.get("schema")
                record_type = record.get("type")
                if record_type == "init":
                    run_index += 1
                    explicit = record.get("run_id") or ""
                    # M26 (2026-09-29): the synthetic id used to use only
                    # the basename, so two runs of identically named logs
                    # in different directories collided and were hard-
                    # rejected as duplicate runs; the full log path keeps
                    # the id stable and directory-distinguishing.
                    run_id = explicit or f"{log_path}#run{run_index}"
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
                elif record_type == "consistency":
                    current.consistency.append(record)
                # memory_anchor and unknown types are tolerated but unused.
    seen: dict[str, Run] = {}
    for run in runs:
        if run.run_id in seen:
            # POST-02 (2026-09-26): identical summaries included.  The two
            # Run objects used to survive side by side and double-count
            # every downstream row/group; duplicate runs are rejected
            # outright instead of merged.
            raise PostprocessError(
                f"run_id {run.run_id!r} appears more than once across the "
                "input logs (sec.11.4); duplicate runs are rejected rather "
                "than silently merged"
            )
        seen[run.run_id] = run
    for run in runs:
        if not _schema_versions_exact(run.schema_versions):
            raise PostprocessError(
                f"run {run.run_id}: unsupported or mixed metric schema "
                f"versions {sorted(run.schema_versions)} (sec.11.4); this "
                f"postprocessor supports schema {SUPPORTED_SCHEMA} only"
            )
    return runs


def _load_manifest_sidecars(run: Run) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (metrics_manifest, service manifest.json).

    M-37 (2026-09-30): a declared sidecar that cannot be READ (missing,
    moved, permission) is a structured sec.11.4 failure -- the previous
    OSError->empty-dict degradation silently emptied the run's grouping
    labels (model/dataset/hardware/kv_policy) and let different workloads
    collapse into one automatic normalization group with a clean exit 0.
    Only runs whose init record declares no ``manifest_path`` at all
    (legacy/off logs) come back with empty dicts, and the auto grouping
    path annotates that case explicitly instead of grouping on empty
    labels.  A *corrupt* sidecar was already a structured failure
    (POST-05).

    R5-batch3 (2026-10-01): sidecar candidate chain, isomorphic to
    ``slo_common.load_request_manifest`` -- the run_dir local copy first
    (the official runners copy ``metrics_manifest.json``/``manifest.json``
    into the run dir root at P2 so a run dir is self-contained and
    decoupled from the repo's restore state), then the init record's
    ``manifest_path`` (the generated/ copy).  Same-generation copies hold
    the same bytes (the run_dir copy is made from the generated/ one at
    run start), so values are unchanged whenever both exist; an archived
    run re-postprocessed after ``clean_test_records.sh`` (generated/
    gone) now reads its own copies instead of failing, and a generated/
    dir re-materialized to a *different* generation no longer feeds a
    foreign sidecar to an old run (the run-local copy wins; the join
    gates remain the last-resort generation check).
    """

    metrics_manifest: dict[str, Any] = {}
    service_manifest: dict[str, Any] = {}
    manifest_path = run.init.get("manifest_path")
    if not manifest_path:
        return metrics_manifest, service_manifest
    declared_path = Path(manifest_path)
    # First *existing* candidate wins (same precedence as
    # slo_common.load_request_manifest: run_dir local > init manifest_path).
    # An existing-but-unreadable candidate is still a structured failure,
    # never a silent fall-through to the next candidate.
    candidates = [run.log_path.parent / "metrics_manifest.json",
                  declared_path]
    path = next((candidate for candidate in candidates
                 if candidate.is_file()), None)

    def _require_object(name: str, loaded: Any) -> dict[str, Any]:
        # P2 (2026-09-29): a sidecar whose top level is not an object used
        # to escape as a bare AttributeError on the first ``.get``; it is a
        # structured sec.11.4 failure instead.
        if not isinstance(loaded, dict):
            raise PostprocessError(
                f"run {run.run_id}: {name} must be a JSON object, got "
                f"{type(loaded).__name__} (fail-closed)"
            )
        return loaded

    if path is None:
        raise PostprocessError(
            f"run {run.run_id}: declared metrics manifest cannot be read "
            f"(fail-closed; empty grouping labels would silently collapse "
            f"comparison groups): no candidate exists among "
            f"{[str(candidate) for candidate in candidates]} "
            f"(init manifest_path={declared_path}; the runner copies the "
            f"sidecars into the run dir, and the generated/ copy is "
            f"produced by plan_materializer.py)"
        )
    try:
        # E4（2026-09-30）：sidecar 也走 _parse_json_object 的
        # parse_constant + 递归有限值门——POST-03 的中毒数据防御原本
        # 只盖 [METRIC] 行，sidecar 的 NaN 会以 float 流入分组标签。
        metrics_manifest = _parse_json_object(
            path.read_text(encoding="utf-8"),
            "metrics manifest " + str(path))
    except OSError as error:
        raise PostprocessError(
            f"run {run.run_id}: declared metrics manifest {path} cannot be "
            f"read (fail-closed; empty grouping labels would silently "
            f"collapse comparison groups): {error}"
        ) from None
    except json.JSONDecodeError as error:
        raise PostprocessError(
            f"run {run.run_id}: metrics manifest {path} is corrupt "
            f"(fail-closed): {error}"
        ) from None
    metrics_manifest = _require_object("metrics manifest " + str(path),
                                       metrics_manifest)
    service_candidates = [run.log_path.parent / "manifest.json",
                          declared_path.parent / "manifest.json"]
    service_manifest_path = next(
        (candidate for candidate in service_candidates
         if candidate.is_file()), None)
    if service_manifest_path is None:
        raise PostprocessError(
            f"run {run.run_id}: service manifest cannot be read "
            f"(fail-closed; empty grouping labels would silently collapse "
            f"comparison groups): no candidate exists among "
            f"{[str(candidate) for candidate in service_candidates]} "
            f"(sibling manifest.json of the init manifest_path and the "
            f"runner's run_dir copy)"
        )
    try:
        service_manifest = _parse_json_object(
            service_manifest_path.read_text(encoding="utf-8"),
            "service manifest " + str(service_manifest_path))
    except OSError as error:
        raise PostprocessError(
            f"run {run.run_id}: service manifest "
            f"{service_manifest_path} cannot be read (fail-closed; empty "
            f"grouping labels would silently collapse comparison "
            f"groups): {error}"
        ) from None
    except json.JSONDecodeError as error:
        raise PostprocessError(
            f"run {run.run_id}: service manifest "
            f"{service_manifest_path} is corrupt (fail-closed): "
            f"{error}"
        ) from None
    service_manifest = _require_object(
        f"service manifest {service_manifest_path}", service_manifest
    )
    return metrics_manifest, service_manifest


def _check_request_completion(run: Run) -> None:
    """sec.11.4: an incomplete request must never be silently counted.

    The empty-requests branch is keyed on the init record's *declared*
    detail level (M6-03, 2026-09-26), not on "the list happens to be
    empty": a run declaring ``detail_level=full`` whose request records
    went missing (truncated log) must fail closed instead of passing with
    a vacuous cross-check.
    """

    summary = run.summary
    if not run.requests:
        if run.init.get("detail_level") == "full":
            raise PostprocessError(
                f"run {run.run_id}: init declares detail_level=full but the "
                "log carries no type=request records; the summary counts "
                "cannot be cross-checked (sec.11.4)"
            )
        # off/summary detail level (or a legacy init without the field):
        # no type=request records exist by construction, nothing to
        # cross-check against.
        return
    completed_records = sum(1 for record in run.requests if record.get("completed"))
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


def _check_consistency(runs: Sequence[Run]) -> None:
    """sec.12.6 fail-closed gate on the collector's consistency verdict.

    Every ``type=consistency`` record carries the collector's own verdict
    (``ok`` / ``violations`` / ``dropped_events``).  A run that is not clean
    aborts here — before raw/normalized/request_metrics are written — so a
    violating run can never exit 0 with products (the verdict is consumed,
    never silently dropped).  A run that carries no consistency record at
    all fails the same way (POST-01/F4/M6-01, 2026-09-26): the collector
    emits one at every run end (MetricCollector.cc finalize), so absence
    means a truncated log whose verdict was never observed — evidence
    absence is a failure, never a pass.
    ``memory_actions_unresolved`` is deliberately not a failure condition:
    the collector reports it for runs with incomplete requests and keeps it
    out of ``ok``.
    """

    for run in runs:
        if not run.consistency:
            raise PostprocessError(
                f"run {run.run_id}: no type=consistency record found; the "
                "collector emits one at every run end, so the log is "
                "truncated or the run never finished (sec.12.6 fail-closed)"
            )
        for record in run.consistency:
            violations = record.get("violations") or []
            dropped = record.get("dropped_events") or 0
            if record.get("ok") is not True or violations or dropped:
                raise PostprocessError(
                    f"run {run.run_id}: collector consistency verdict is "
                    f"not clean (ok={record.get('ok')!r}, "
                    f"dropped_events={dropped}): "
                    f"violations="
                    f"{json.dumps(violations, ensure_ascii=False)[:400]}"
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
# NA / FIRST_TOKEN_SOURCE_* / TERMINAL_STATUS_* 闭集枚举自 metrics_schema
# 导入(G7-2 修复 2026-09-27:枚举单一事实源移入 frozen schema 模块,
# 写入点与测试契约同源;原本模块字面量定义删除,消费方走 metrics_schema
# 或本模块再导出的同名)。


def _int_or_na(value: Any, *, run_id: str, queue_index: Any,
               column: str) -> Any:
    """Render a metrics JSON integer, or the frozen NA placeholder.

    Tightened (Minor, 2026-09-30): booleans are excluded like
    ``_int_or_none``/``_require_join_int`` (P2, 2026-09-29) and any
    non-integer value (float such as 1.5 used to be silently truncated by
    ``int(...)``; numeric strings were silently accepted) is now a
    structured sec.11.4 failure -- the C++ collector only ever emits
    integers or null into these timing columns.
    """
    if value is None or isinstance(value, bool):
        return NA
    if not isinstance(value, int):
        raise PostprocessError(
            f"run {run_id}: queue_index {queue_index}: {column}={value!r} "
            "is not an integer (fail-closed timing column)"
        )
    return value


def _int_or_none(value: Any) -> Optional[int]:
    """Metrics JSON integer or None (WP9 proxy arithmetic helper).

    E2（2026-09-30）：与 _int_or_na 同口径的严格 int 门——数字字符串
    （"32"）、浮点（12.0，int() 截断）不再被静默收编，非 int 一律按
    缺席（None）处理。旧实现会让损坏 manifest 的字符串数字被"修好"后
    写进冻结产品列而不是暴露为 NA。"""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _require_join_int(record: dict[str, Any], key: str, default: Any,
                      run: "Run", queue_index: Any) -> int:
    """Required join-column integer with a structured failure (POST-05,
    2026-09-26): ``int(record.get(key, default))`` used to escape as a bare
    TypeError on an explicit ``null`` / non-numeric value instead of the
    sec.11.4 PostprocessError every other join failure raises."""

    value = record.get(key, default)
    if isinstance(value, bool):
        value = None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise PostprocessError(
            f"run {run.run_id}: queue_index {queue_index} request "
            f"{record.get('request_id')!r}: {key} must be an integer, got "
            f"{value!r} (fail-closed join column)"
        ) from None


def _manifest_request_entries(
    run: Run,
    metrics_manifest: dict[str, Any],
    service_manifest: dict[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    """Per-request manifest entries keyed by queue_index (fail-closed join).

    Merges the run's ``metrics_manifest.json`` (the C++ collector input)
    with its sibling service ``manifest.json``.  Both must agree on
    ``request_id`` whenever they carry it for the same ``queue_index``;
    fields absent from one manifest are taken from the other.  The merge
    exists only ACROSS the two sidecars: a ``queue_index`` repeated inside
    one sidecar is a manifest defect and is rejected (POST-05, 2026-09-26)
    instead of being silently field-merged.  Returns
    ``(entries_by_queue_index, per_manifest_request_counts)``.
    """
    merged: dict[int, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for name, manifest in (
        ("metrics_manifest.json", metrics_manifest),
        ("manifest.json", service_manifest),
    ):
        requests = manifest.get("requests")
        if not isinstance(requests, list):
            counts[name] = 0
            continue
        counts[name] = len(requests)
        seen_in_manifest: set[int] = set()
        for entry in requests:
            if not isinstance(entry, dict) or "queue_index" not in entry:
                raise PostprocessError(
                    f"run {run.run_id}: {name} request entry without "
                    "queue_index cannot be joined to the request records"
                )
            queue_index = entry["queue_index"]
            if queue_index in seen_in_manifest:
                raise PostprocessError(
                    f"run {run.run_id}: {name} declares queue_index "
                    f"{queue_index} more than once; a duplicate manifest "
                    "entry cannot be joined unambiguously (fail-closed)"
                )
            seen_in_manifest.add(queue_index)
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


def _invariant_stage_int(
    run_id: str, record: dict[str, Any], column: str, queue_index: Any
) -> Optional[int]:
    """E1（2026-09-30）：不变量检查入口的整数门。

    旧实现在原始 JSON 值上直接比较（``value < 0`` / ``earlier >
    later`` / ``sum(parts)``），一行损坏数据（如 ``{"queue_ns": "slow"}``）
    会以裸 ``TypeError`` 逃过 main() 只捕 PostprocessError 的结构化失败
    面板；同模块整改史（POST-04/P2）明确把这类逃逸定为缺陷。此处先做
    与 _int_or_na 同款的严格 int 门（拒 bool/字符串/浮点），非 int 即
    结构化失败。"""
    value = record.get(column)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise PostprocessError(
            f"run {run_id}: queue_index {queue_index}: {column} "
            f"{value!r} is not an integer (stage/boundary columns are "
            "integer nanoseconds; fail-closed)"
        )
    return value


def _check_request_row_invariants(
    run_id: str, record: dict[str, Any], queue_index: Any
) -> None:
    """Ordering/decomposition invariants on rows that carry values.

    Checked only among the fields that are present: the stage durations are
    non-negative (a negative queue_ns used to slip into the frozen product
    columns unchecked), the boundary chain arrival <= prefill_start <=
    prefill_end <= decode_start <= completion is monotone (the same chain
    the C++ collector checks on completed requests), arrival <= first_token
    <= completion (first_token skipped while NA), e2e == completion -
    arrival, and queue+prefill+gap+decode == e2e.  Any violation aborts
    (no silent clamping).
    """
    # Stage durations: non-negative whenever present, independent of the
    # arrival/completion presence guard below (a truncated row must not
    # smuggle a negative stage into the frozen columns either).
    for column in ("queue_ns", "prefill_ns", "prefill_decode_gap_ns",
                   "decode_ns"):
        value = _invariant_stage_int(run_id, record, column, queue_index)
        if value is not None and value < 0:
            raise PostprocessError(
                f"run {run_id}: queue_index {queue_index}: {column} "
                f"{value} is negative (stage durations are non-negative)"
            )
    # Boundary chain monotonicity, pairwise among the present fields.
    chain = (
        ("arrival_ns", "prefill_start_ns"),
        ("prefill_start_ns", "prefill_end_ns"),
        ("prefill_end_ns", "decode_start_ns"),
        ("decode_start_ns", "completion_ns"),
    )
    for earlier_column, later_column in chain:
        earlier = _invariant_stage_int(
            run_id, record, earlier_column, queue_index)
        later = _invariant_stage_int(
            run_id, record, later_column, queue_index)
        if earlier is not None and later is not None and earlier > later:
            raise PostprocessError(
                f"run {run_id}: queue_index {queue_index}: "
                f"{earlier_column} {earlier} > {later_column} {later} "
                "(stage boundaries must be monotone)"
            )
    arrival = _invariant_stage_int(run_id, record, "arrival_ns", queue_index)
    first_token = _invariant_stage_int(
        run_id, record, "first_token_ns", queue_index)
    completion = _invariant_stage_int(
        run_id, record, "completion_ns", queue_index)
    e2e = _invariant_stage_int(run_id, record, "e2e_ns", queue_index)
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
        _invariant_stage_int(run_id, record, column, queue_index)
        for column in ("queue_ns", "prefill_ns", "prefill_decode_gap_ns",
                       "decode_ns")
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
# B3_W 60s gate-2 (decision equivalence) failed -- the split's physical
# perturbation is amplified by the closed loop (see
# /tmp/slo_wps/gates/B3_W.DONE, evidence /tmp/slo_wps/b3/W/) -- so the
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
# first_train_end_ns -- wscllm-specific structure: trains are DECODE-ONLY
# (PD separation: the P side emits whole prefill segments with no train
# concept, WP9_CONTRACT sec.2), so every ledger row is a D-instance train.
# The ledger row's tick is the emission (start) boundary; a decode instance
# is busy until the train's end barrier completes (busy gate = one train in
# flight, in_flight_train, wsc_llm_online_scheduler.py sec.3.6 D role), so
# the NEXT ledger row on the same instance is the first decision boundary
# after the end barrier and is used as the train-end tick (last train of an
# instance -> proxy not obtainable -> NA).  Sentinel trains (T_max-capped
# trains without exit members; the majority of rows since decode_length=64
# rides multiple 8-iteration trains) are ordinary ledger rows -- a debut
# joining a sentinel train is handled identically.
# Degenerate 1-iteration trains (N=1 -> share=1) make the proxy equal that
# next-emission boundary, which sits after the barrier tick recorded as
# completion; such values are clamped to completion (the first token
# physically cannot complete after the request) and flagged
# "clamped_to_completion" in instructions.
#
# W_bytes / KV bytes per token: the repository's authoritative conversions
# (wsc_llm_scheduler.estimate_model_weight_bytes / kv_cache_bytes_for_
# tokens -- the wscllm twin of face_scheduler's frozen formulas, weights
# read once per iteration), parameterised by the trace_config.csv model
# rows; no value is invented here.
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


def _load_model_bytes(
    run_log_dir: Optional[Path] = None,
) -> tuple[Optional[int], Optional[int]]:
    """(W_bytes, kv_bytes_per_token) from this workload's trace_config.csv.

    R2-3 (2026-09-30): run-local copies take precedence over the repo
    canonical file, mirroring hbm_watermark's resolve_trace_config_path
    chain (run_dir/trace_config.csv > run_dir/trace_config.csv.snapshot >
    repo default) -- a run produced with a locally modified model row used
    to get proxy weights silently computed from the stale canonical model.

    Returns (None, None) when the config or a model row is missing/invalid
    (proxy then degrades to NA; nothing is fabricated).
    """

    try:
        from wsc_llm_scheduler import (
            WscLlmModel,
            estimate_model_weight_bytes,
            kv_cache_bytes_for_tokens,
        )
    except ImportError:
        return None, None
    config_path = MODULE_DIR / "trace_config.csv"
    if run_log_dir is not None:
        for candidate in (run_log_dir / "trace_config.csv",
                          run_log_dir / "trace_config.csv.snapshot"):
            if candidate.is_file():
                config_path = candidate
                break
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
        model = WscLlmModel(**model_kwargs)
        return estimate_model_weight_bytes(model), kv_cache_bytes_for_tokens(model, 1)
    except ValueError:
        return None, None


class _TrainProxyIndex:
    """Lazy per-run index over results/train_ledger.jsonl (read-only)."""

    def __init__(self, run: Run):
        self._run = run
        # R2-3: run-local trace_config candidate dir for the proxy weights
        # (the run's metrics log lives in the run's results/ dir).
        self.run_log_dir = run.log_path.parent
        self._loaded = False
        # request_id -> (ledger_row_position, row) of its first (non
        # first_step) train carrying it in joiners.
        self.first_train_by_request: dict[str, tuple[int, dict[str, Any]]] = {}
        # ledger row position -> tick of the next row on the same instance.
        self.next_tick_by_position: dict[int, int] = {}

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
                for line_number, line in enumerate(source, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        # POST-05 (2026-09-26): one corrupt line used to
                        # abort the whole load and silently degrade every
                        # first-token proxy to NA; a damaged ledger is a
                        # structured failure instead.
                        raise PostprocessError(
                            f"run {self._run.run_id}: train ledger "
                            f"{ledger_path}:{line_number} is corrupt "
                            f"(fail-closed): {error}"
                        ) from None
                    # E3（2026-09-30）：合法 JSON 的标量/数组行同为形状
                    # 损坏——旧行为静默吞掉并把对应 proxy 判成
                    # no_train_ledger_joiner_row 类 NA（观测口径被静默
                    # 改变），与 POST-05 的结构化失败口径对齐。
                    if not isinstance(record, dict):
                        raise PostprocessError(
                            f"run {self._run.run_id}: train ledger "
                            f"{ledger_path}:{line_number} row is not a "
                            f"JSON object (fail-closed): "
                            f"{type(record).__name__}"
                        ) from None
                    rows.append(record)
        except OSError as error:
            # P2 (2026-09-29): an OSError used to silently degrade to the
            # same path as a *missing* ledger, making the run-level
            # "proxy_unavailable:no_train_ledger(results/)" note a lie; a
            # ledger that exists but cannot be read is a structured
            # failure (str(OSError) carries errno and the path).
            raise PostprocessError(
                f"run {self._run.run_id}: train ledger {ledger_path} exists "
                f"but cannot be read (fail-closed): {error}"
            ) from None
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
            if isinstance(tick, int) and not isinstance(tick, bool):
                next_by_instance[instance] = tick
        self._available = True


_PROXY_MODEL_BYTES: dict[str, tuple[Optional[int], Optional[int]]] = {}


def _model_bytes_cached(
    run_log_dir: Optional[Path] = None,
) -> tuple[Optional[int], Optional[int]]:
    key = str(run_log_dir) if run_log_dir is not None else "<canonical>"
    if key not in _PROXY_MODEL_BYTES:
        _PROXY_MODEL_BYTES[key] = _load_model_bytes(run_log_dir)
    return _PROXY_MODEL_BYTES[key]


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
    weight_bytes, kv_per_token = _model_bytes_cached(
        getattr(index, "run_log_dir", None))
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

    Timing/stage columns come from the C++ ``type=request`` records;
    ``request_type``/length columns come from the joined manifest entries
    (WP2 passthrough) and degrade to ``unknown``/``NA`` with a per-row
    ``instructions`` note -- never a guess.
    """
    rows: list[dict[str, Any]] = []
    # WP9 fallback (B4): train-interpolated proxy index over the run's
    # train_ledger.jsonl (loaded lazily; absent ledger -> NA with a note).
    proxy_index = _TrainProxyIndex(run)
    proxy_available = proxy_index.available()
    for record in run.requests:
        queue_index = record.get("queue_index")
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
        completed = bool(record.get("completed"))
        terminal_status = (
            TERMINAL_STATUS_COMPLETED if completed
            else TERMINAL_STATUS_FAILED)
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
            notes.append("proxy_unavailable:no_train_ledger(results/)")
        # P2 (2026-09-29): the C++ collector emits ``first_token_note`` on
        # the request record when an ORDER violation nulled first_token_ns
        # (or the event is absent / the manifest decode length missing --
        # MetricCollector.cc); the postprocessor used to drop the field
        # silently.  It surfaces through the same notes mechanism as the
        # proxy backfill above.
        first_token_note = record.get("first_token_note")
        if first_token_note:
            notes.append(f"first_token_note={first_token_note}")
        if first_token_ns is not None:
            # Ordering invariant is re-checked fail-closed on the FILLED
            # value (exact or proxy): arrival <= first_token <= completion.
            checked = dict(record)
            checked["first_token_ns"] = first_token_ns
            _check_request_row_invariants(run.run_id, checked, queue_index)
        else:
            _check_request_row_invariants(run.run_id, record, queue_index)
        # Hard invariant (dl1, 2026-09-05): a train_interpolated row with
        # decode_length==1 must carry first_token_ns == completion_ns -- the
        # pin/clamp in _first_token_proxy_value guarantees it, so a mismatch
        # means that path broke (fail-closed, the silent-early lesson).
        # exact rows are EXEMPT: WP9_CONTRACT §6 relaxes dl1 equality to an
        # informational TP skew (first_token <= completion is legal).
        completion_ns = _int_or_none(record.get("completion_ns"))
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
            else:
                parsed = _int_or_none(value)
                if parsed is None:
                    raise PostprocessError(
                        f"run {run.run_id}: queue_index {queue_index}: "
                        f"manifest field {column}={value!r} is not an "
                        "integer (fail-closed)"
                    )
                lengths[column] = parsed
        # P2 (2026-09-29): the dl1 gate reads the SAME parsed decode_length
        # the proxy uses (_int_or_none of the manifest entry) -- the raw
        # manifest value let a string "1" bypass the invariant.
        if (first_token_source == FIRST_TOKEN_SOURCE_TRAIN_INTERPOLATED
                and first_token_ns is not None
                and completion_ns is not None
                and lengths["decode_length"] == 1
                and first_token_ns != completion_ns):
            raise PostprocessError(
                f"run {run.run_id}: queue_index {queue_index} request "
                f"{record.get('request_id')!r}: decode_length==1 proxy "
                "first_token_ns != completion_ns (dl1 first token IS the "
                "last token; pinned_to_completion path broken)"
            )
        rows.append(
            {
                "queue_index": int(queue_index),
                "request_id": record.get("request_id", ""),
                "session_id": record.get("session_id", ""),
                "turn_index": _require_join_int(
                    record, "turn_index", 0, run, queue_index),
                "request_type": request_type,
                "terminal_status": terminal_status,
                "arrival_ns": _int_or_na(
                    record.get("arrival_ns"), run_id=run.run_id,
                    queue_index=queue_index, column="arrival_ns"),
                "prefill_start_ns": _int_or_na(
                    record.get("prefill_start_ns"), run_id=run.run_id,
                    queue_index=queue_index, column="prefill_start_ns"),
                "prefill_end_ns": _int_or_na(
                    record.get("prefill_end_ns"), run_id=run.run_id,
                    queue_index=queue_index, column="prefill_end_ns"),
                "decode_start_ns": _int_or_na(
                    record.get("decode_start_ns"), run_id=run.run_id,
                    queue_index=queue_index, column="decode_start_ns"),
                # WP9 首 token(2026-08-26):exact = request 记录的
                # first_token_ns(事件码 8,split ON 时由首步批 first_token
                # 标记产出);缺失时 train-interpolated proxy(B4 fallback,
                # display-only;见 _TrainProxyIndex 块);均无 → NA。
                "first_token_ns": (
                    NA if first_token_ns is None else first_token_ns),
                "first_token_source": (
                    first_token_source if first_token_source is not None
                    else NA),
                "completion_ns": _int_or_na(
                    record.get("completion_ns"), run_id=run.run_id,
                    queue_index=queue_index, column="completion_ns"),
                "queue_ns": _int_or_na(
                    record.get("queue_ns"), run_id=run.run_id,
                    queue_index=queue_index, column="queue_ns"),
                "prefill_ns": _int_or_na(
                    record.get("prefill_ns"), run_id=run.run_id,
                    queue_index=queue_index, column="prefill_ns"),
                "prefill_decode_gap_ns": _int_or_na(
                    record.get("prefill_decode_gap_ns"), run_id=run.run_id,
                    queue_index=queue_index, column="prefill_decode_gap_ns"
                ),
                "decode_ns": _int_or_na(
                    record.get("decode_ns"), run_id=run.run_id,
                    queue_index=queue_index, column="decode_ns"),
                "e2e_ns": _int_or_na(
                    record.get("e2e_ns"), run_id=run.run_id,
                    queue_index=queue_index, column="e2e_ns"),
                # KV-hit/restore columns stay NA in this product: they are
                # filled by the slo_tools kv_cache_adapter's canonical
                # cache_events.csv (WP4) and restore decomposition (WP5),
                # not by this postprocessor.
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


def build_request_metrics(
    runs: Sequence[Run],
) -> tuple[Optional[list[dict[str, Any]]], dict[str, Any]]:
    """Build the frozen-schema request_metrics.csv row set in memory.

    Full detail only: summary/off logs carry no ``type=request`` records,
    so ``(None, skip-summary)`` is returned and the skip reason lands in
    the run's postprocess.log.  Any join failure aborts with a structured
    sec.11.4 error BEFORE any product is written (M21, 2026-09-29).

    S4 (2026-09-29): the join is checked as a whole key set, not only per
    row -- a ``queue_index`` repeated among the log's request records, and
    any request-record/manifest key-set mismatch on either side, fail
    closed with the concrete offending keys.  (manifest={0,1,2} vs log
    records={0,1,1} used to pass the 3==3 count check and freeze a
    duplicate row with a lost row into the product.)
    """
    with_requests = [run for run in runs if run.requests]
    if not with_requests:
        detail_levels = sorted(str(run.init.get("detail_level")) for run in runs)
        summary = {
            "request_metrics": "skipped",
            "reason": (
                "no type=request [METRIC] records in the input logs "
                "(request rows are emitted at --metrics-detail=full only)"
            ),
            "detail_levels": detail_levels,
        }
        return None, summary
    if len(with_requests) > 1:
        raise PostprocessError(
            "request_metrics.csv is a single-run product; "
            f"{len(with_requests)} runs carry request records "
            f"({', '.join(run.run_id for run in with_requests)}); "
            "postprocess one run log at a time"
        )
    run = with_requests[0]
    if not run.init.get("manifest_path"):
        raise PostprocessError(
            f"run {run.run_id}: request records cannot be joined without "
            "the init record manifest_path (fail-closed)"
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
    # S4: the log's own queue_index set must be duplicate-free -- a
    # repeated record cannot be joined unambiguously even when the count
    # check passes (duplicate row + lost row keep the total unchanged).
    seen_keys: set[Any] = set()
    duplicate_keys: list[Any] = []
    for record in run.requests:
        key = record.get("queue_index")
        if key in seen_keys:
            duplicate_keys.append(key)
        seen_keys.add(key)
    if duplicate_keys:
        raise PostprocessError(
            f"run {run.run_id}: the log declares queue_index "
            f"{duplicate_keys!r} more than once; a duplicate request "
            "record cannot be joined unambiguously (fail-closed; rows are "
            "never skipped or truncated)"
        )
    # S4: key-set-level bidirectional check -- both set differences must
    # be empty, with the concrete missing/extra keys in the failure.
    manifest_keys = set(entries)
    absent_from_log = sorted(manifest_keys - seen_keys, key=repr)
    absent_from_manifest = sorted(seen_keys - manifest_keys, key=repr)
    if absent_from_log or absent_from_manifest:
        raise PostprocessError(
            f"run {run.run_id}: request-record/manifest queue_index key "
            "sets differ (fail-closed; rows are never skipped or "
            f"truncated): manifest keys with no log request record "
            f"{absent_from_log!r}; log request records with no manifest "
            f"key {absent_from_manifest!r}"
        )
    rows = _request_metric_rows(run, entries)
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
        "request_rows": len(rows),
        "terminal_status_counts": status_counts,
        "request_type_counts": type_counts,
        "first_token_source_counts": source_counts,
        "rows_with_notes": sum(1 for row in rows if row["instructions"]),
    }
    return rows, summary


def write_request_metrics(
    runs: Sequence[Run], path: Path
) -> dict[str, Any]:
    """Write the frozen-schema request_metrics.csv next to the raw CSV.

    Single-run entry point kept for direct callers; main() stages all
    three products together.  The row set is built fully in memory first
    (M21) and the file is materialised atomically by ``_write_csv``.

    Positioning (dead-code audit, 2026-09-30): this entry point has no
    in-repo caller -- main() owns the product writing in official runs.
    Kept (not deleted) as the documented single-run API for direct
    callers; deletion is a cross-repo ruling.
    """
    rows, summary = build_request_metrics(runs)
    if rows is not None:
        _write_csv(path, REQUEST_METRICS_COLUMNS, rows)
    print(json.dumps({"out_requests": str(path), **summary}))
    return summary


def _object_field(value: Any, description: str, run: "Run") -> dict[str, Any]:
    """JSON sub-object with a structured failure (M20, 2026-09-29):
    ``x.get("k", {}).get(...)`` escaped as a bare AttributeError on an
    explicit ``null`` -- the ``{}`` default only covers a *missing* key.
    ``null`` reads as absent; any other non-object value is a sec.11.4
    failure."""

    if value is None:
        return {}
    if not isinstance(value, dict):
        raise PostprocessError(
            f"run {run.run_id}: {description} must be an object, got "
            f"{type(value).__name__} (fail-closed)"
        )
    return value


def _manifest_block(manifest: dict[str, Any], key: str,
                    run: "Run") -> dict[str, Any]:
    """Service-manifest sub-object with a structured failure (POST-05,
    2026-09-26): ``manifest.get(key, {}).get(...)`` used to escape as a bare
    AttributeError when the block held a non-object JSON value."""

    block = manifest.get(key, {})
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise PostprocessError(
            f"run {run.run_id}: service manifest block {key!r} must be an "
            f"object, got {type(block).__name__} (fail-closed)"
        )
    return block


def _labels(run: Run, service_manifest: dict[str, Any], run_config: dict[str, Any]) -> dict[str, str]:
    model = run_config.get("model") or _manifest_block(
        service_manifest, "model", run).get("name", "")
    hardware_block = _manifest_block(service_manifest, "hardware", run)
    hardware = (
        run_config.get("hardware")
        or hardware_block.get("label")
        or hardware_block.get("case")
        or hardware_block.get("capacity_profile")
        or ""
    )
    kv_policy = run_config.get("kv_policy") or _manifest_block(
        service_manifest, "kv_management", run).get("policy", "")
    if kv_policy and str(kv_policy) not in KV_POLICY_LABELS:
        # A.5/2026-09-05：kv_policy label 枚举校验；B2/2026-09 加
        # session_lru_tiered，收尾后唯一合法值为 session_lru_tiered
        # （与上方 KV_POLICY_LABELS 同源）；空串=未标注仍放行
        # （既有产物兼容），非空未知值 fail-closed。
        raise PostprocessError(
            f"kv_policy label {kv_policy!r} not in the known set "
            f"{sorted(KV_POLICY_LABELS)} (typo would silently split "
            "comparison groups)")
    dataset = run_config.get("dataset") or ""
    if not dataset:
        queue_csv = service_manifest.get("request_queue_csv", "")
        # M20 (2026-09-29): a non-string request_queue_csv used to escape
        # as a bare TypeError from Path(...); structured failure instead.
        if queue_csv is None:
            queue_csv = ""
        if not isinstance(queue_csv, str):
            raise PostprocessError(
                f"run {run.run_id}: service manifest request_queue_csv "
                f"must be a string, got {type(queue_csv).__name__} "
                "(fail-closed)"
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


def _microbench_rows(
    run: Run,
    labels: dict[str, str],
    metrics_manifest: dict[str, Any],
    notes: list[str],
) -> list[dict[str, Any]]:
    """Microbenchmark point rows (run_mode=microbenchmark).

    Positioning (dead-code audit, 2026-09-30): this repo has NO producer
    for microbenchmark runs -- the C++ collector emits run_mode=service
    only and plan_materializer synthesizes service manifests only, so the
    whole microbenchmark path (this builder plus the
    simulator_microbenchmark source rows) is dead in production here.
    Kept (not deleted) per cross-repo ruling: the frozen schema and the
    run_mode vocabulary still declare the mode.  Never reachable from
    official run logs today.
    """
    point = _object_field(
        metrics_manifest.get("microbenchmark"),
        "metrics manifest microbenchmark block", run,
    )
    # Fail-closed: a point that declares tp_degree without ranks gives no
    # attribution for the utilization denominators (the numerator would
    # aggregate every rank_compute record while the denominator still
    # assumes tp_degree ranks, silently inflating the ratios).
    tp_degree = point.get("tp_degree", "")
    if tp_degree and "ranks" not in point:
        raise PostprocessError(
            f"run {run.run_id}: microbenchmark manifest point declares "
            f"tp_degree {tp_degree!r} but no ranks; the utilization "
            "denominators cannot be attributed (fail-closed)"
        )
    if tp_degree == 0:
        # P2 (2026-09-29): tp_degree=0 used to disable the utilization
        # columns silently; the disabling is now explicit in the
        # postprocess summary notes.
        notes.append(
            f"run {run.run_id}: utilization disabled: tp_degree=0 "
            "(microbenchmark manifest point declares tp_degree=0; "
            "compute_util/hbm_bw_util left empty)"
        )
    active_ranks = set(point.get("ranks", []))
    ops_by_rank: dict[int, int] = {}
    bytes_by_rank: dict[int, int] = {}
    peak_flops: Optional[float] = None
    local_bw: Optional[float] = None
    for record in run.rank_compute:
        rank = record.get("rank")
        if active_ranks and rank not in active_ranks:
            continue
        ops_by_rank[rank] = int(record.get("total_num_ops", 0))
        bytes_by_rank[rank] = int(record.get("total_local_bytes", 0))
        # First active rank wins (the C++ iteration record takes its
        # denominators from the first active rank's system config,
        # reference_sys in MetricCollector.cc); a later rank must not
        # silently replace it on heterogeneous per-rank hardware.
        if peak_flops is None:
            peak_flops = record.get("peak_flops_per_second") or None
        if local_bw is None:
            local_bw = record.get("local_hbm_bw_bytes_per_second") or None
    rows = []
    for iteration in run.iterations:
        row = _base_row(run, labels, SOURCE_MICROBENCHMARK)
        iteration_time_ns = iteration.get("iteration_time_ns")
        total_ops = sum(ops_by_rank.values())
        total_bytes = sum(bytes_by_rank.values())
        compute_util: Optional[float] = None
        hbm_bw_util: Optional[float] = None
        if (
            iteration_time_ns
            and tp_degree
            and peak_flops
            and local_bw
        ):
            window_s = iteration_time_ns / 1e9
            # doc sec.8.7, recomputed from the rank_compute records.  The
            # C++ iteration record carries its own compute_util/hbm_bw_util
            # fields with the same formula (num_ops_total / (tp_degree *
            # peak_flops_per_second * iteration_time_seconds), denominators
            # from the FIRST active rank's system config), but the C++
            # collector emits them only when the roofline gate is on
            # (nullptr otherwise -- MetricCollector.cc); this recomputation
            # is the roofline-independent port of that contract.
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
    return rows


def _value_or_na(value: Any) -> Any:
    """Render a JSON null metric cell as the frozen NA placeholder.

    M-20 (2026-09-30): the collector emits ``null`` capacity/utilization
    cells (with a ``timeavg_util_note`` in the log) when a rank has no
    replayed planner ledger, so the frozen column stays explicitly
    "unavailable" here too -- an empty cell (or a fabricated 0) would read
    as a measured value.
    """
    return NA if value is None else value


def _ledger_rank_or_fail(run: Run, record: dict[str, Any],
                         record_kind: str) -> int:
    """Required integer rank of a planner-memory record (fail-closed).

    A non-integer rank used to flow into the rank-keyed dicts and blow up
    only later at the sorted() merge (bare TypeError on mixed key types).
    """
    rank = record.get("rank")
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise PostprocessError(
            f"run {run.run_id}: {record_kind} rank must be an integer, got "
            f"{rank!r} (fail-closed)"
        )
    return rank


def _memory_ledger_rows(run: Run, labels: dict[str, str]) -> list[dict[str, Any]]:
    def _peaks_only_row(rank: Any, peak: dict[str, Any]) -> dict[str, Any]:
        """Peaks-only ledger row (rank with planner peaks but no
        capacity_timeavg record; timeavg/action columns stay empty)."""
        ledger = _object_field(
            peak.get("ledger"),
            f"planner_memory_peaks rank {rank!r} ledger", run,
        )
        physical = _object_field(
            peak.get("peak_physical"),
            f"planner_memory_peaks rank {rank!r} peak_physical", run,
        )
        committed = _object_field(
            peak.get("peak_committed"),
            f"planner_memory_peaks rank {rank!r} peak_committed", run,
        )
        row = _base_row(run, labels, SOURCE_PLANNER_MEMORY)
        row.update(
            {
                "rank": rank,
                # M-20 口径对齐（2026-10-01）：capacity 缺席渲染 NA（不伪
                # 造空串），与 timeavg 路径同款。
                "capacity_bytes": _value_or_na(
                    ledger.get("capacity_bytes", NA)),
                "peak_physical_used_bytes": physical.get(
                    "peak_value_bytes", ""
                ),
                "peak_committed_used_bytes": committed.get(
                    "peak_value_bytes", ""
                ),
            }
        )
        return row

    peaks_by_rank: dict[Any, dict[str, Any]] = {}
    for record in run.planner_peaks:
        rank = _ledger_rank_or_fail(run, record, "planner_memory_peaks")
        if rank in peaks_by_rank:
            # P2 (2026-09-29): a repeated planner_memory_peaks rank used to
            # be silently last-write-wins in this dict; duplicates are
            # rejected like every other ambiguous join input.
            raise PostprocessError(
                f"run {run.run_id}: planner_memory_peaks declares rank "
                f"{rank!r} more than once; duplicate peak records cannot "
                "be attributed unambiguously (fail-closed)"
            )
        peaks_by_rank[rank] = record
    rows = []
    seen_timeavg_ranks: set[Any] = set()
    for record in run.capacity_timeavg:
        rank = _ledger_rank_or_fail(run, record, "capacity_timeavg")
        if rank in seen_timeavg_ranks:
            # Minor gate (2026-09-30): duplicate capacity_timeavg ranks get
            # the same fail-closed treatment as duplicate planner peak
            # ranks -- the second row silently duplicated the rank's
            # weight in any downstream aggregation.
            raise PostprocessError(
                f"run {run.run_id}: capacity_timeavg declares rank "
                f"{rank!r} more than once; duplicate records cannot be "
                "attributed unambiguously (fail-closed)"
            )
        seen_timeavg_ranks.add(rank)
        row = _base_row(run, labels, SOURCE_PLANNER_MEMORY)
        row.update(
            {
                "rank": rank,
                "capacity_bytes": _value_or_na(record.get("capacity_bytes")),
                "resident_capacity_timeavg_util": _value_or_na(
                    record.get("resident_capacity_timeavg_util")
                ),
                "committed_capacity_timeavg_util": _value_or_na(
                    record.get("committed_capacity_timeavg_util")
                ),
                "memory_actions_total": record.get("actions_total", ""),
                "memory_actions_replayed": record.get("actions_replayed", ""),
                "memory_actions_unresolved": record.get("actions_unresolved", ""),
            }
        )
        peak = peaks_by_rank.get(rank)
        if peak is not None:
            physical = _object_field(
                peak.get("peak_physical"),
                f"planner_memory_peaks rank {rank!r} peak_physical", run,
            )
            committed = _object_field(
                peak.get("peak_committed"),
                f"planner_memory_peaks rank {rank!r} peak_committed", run,
            )
            row["peak_physical_used_bytes"] = physical.get(
                "peak_value_bytes", ""
            )
            row["peak_committed_used_bytes"] = committed.get(
                "peak_value_bytes", ""
            )
        rows.append(row)
    if not rows:
        # Peaks without replay (e.g. microbenchmark runs) still surface.
        for rank, peak in sorted(peaks_by_rank.items()):
            rows.append(_peaks_only_row(rank, peak))
    else:
        # Symmetric residual (R5-batch3, 2026-10-01): ranks carrying
        # planner peaks but no capacity_timeavg record (mixed/malformed
        # logs) get the same peaks-only row as the empty branch above --
        # the one-way merge used to drop them silently (no row, no note),
        # against the repo-wide "row shrinkage must leave a trace" rule.
        # Unreachable for current collector products (G2-1: every rank
        # emits capacity_timeavg, zero-activity ranks included), so the
        # production output is byte-identical.
        residual = sorted(
            (rank, peak) for rank, peak in peaks_by_rank.items()
            if rank not in seen_timeavg_ranks
        )
        for rank, peak in residual:
            rows.append(_peaks_only_row(rank, peak))
    return rows


def build_raw_rows(
    runs: Sequence[Run],
    run_configs: dict[str, dict[str, Any]],
    notes: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Raw product rows; structured ``notes`` (utilization disabled, ...)
    are appended to the caller's list when one is passed (P2, 2026-09-29)
    and surface in main()'s summary JSON."""
    notes = notes if notes is not None else []
    rows: list[dict[str, Any]] = []
    for run in runs:
        _check_request_completion(run)
        metrics_manifest, service_manifest = _load_manifest_sidecars(run)
        labels = _labels(run, service_manifest, run_configs.get(run.run_id, {}))
        run_mode = run.init.get("run_mode", "service")
        if run_mode == "microbenchmark":
            rows.extend(_microbench_rows(run, labels, metrics_manifest, notes))
        else:
            rows.extend(_service_rows(run, labels))
        rows.extend(_memory_ledger_rows(run, labels))
    return rows


@dataclass
class Group:
    group_id: str
    runs: list[Run]
    baseline_run_id: Optional[str]
    labels_by_run: dict[str, dict[str, Any]]


def _check_group_digest_gate(
    group_id: str, members: Sequence[Run], notes: Optional[list[str]]
) -> None:
    """sec.11.4 workload digest gate shared by both grouping paths.

    M-37 (2026-09-30): the gate compared ("", "") tuples -- no writer in
    this repository emits the digest keys into init records, so every
    official run passed it vacuously and the pass was recorded nowhere.
    Now a group whose runs declare no digests at all records an explicit
    SKIP note (workload identity is then not digest-verified) instead of a
    silent vacuous pass; a group with conflicting digest tuples still
    fails exactly as before.
    """

    digest_tuples = {
        (
            run.init.get("trace_digest", ""),
            run.init.get("request_mapping_digest", ""),
        )
        for run in members
    }
    if digest_tuples == {("", "")}:
        if notes is not None:
            notes.append(
                f"group {group_id}: digest gate SKIPPED: no run declares "
                "trace_digest/request_mapping_digest (no in-repo writer "
                "emits them into init records); workload identity is not "
                "digest-verified for this group"
            )
        return
    if len(digest_tuples) > 1:
        raise PostprocessError(
            f"group {group_id}: workload digest mismatch across runs "
            "without allow_digest_mismatch (sec.11.4)"
        )


def build_groups(
    runs: Sequence[Run],
    group_config: Optional[dict[str, Any]],
    notes: Optional[list[str]] = None,
) -> tuple[list[Group], dict[str, dict[str, Any]]]:
    """Comparison groups; returns (groups, per-run label overrides).

    ``notes`` (optional) collects structured SKIP annotations (M-37: the
    empty-digest gate skip and the no-provenance run separation) so a
    skipped gate is visible in main()'s summary instead of silently
    passing.
    """

    run_configs: dict[str, dict[str, Any]] = {}
    by_id = {run.run_id: run for run in runs}
    groups: list[Group] = []
    if group_config is None:
        # Default: automatic homogeneous groups keyed by workload identity, so
        # mixed feeds never silently mix different workloads (sec.11.4).
        # M22 (2026-09-29): the key used run.init trace_digest/
        # request_mapping_digest, but no manifest writer in this repository
        # emits those keys, so it collapsed to (repo_variant, run_mode) and
        # different workloads silently became each other's normalization
        # denominators.  Group by the workload fields the manifests actually
        # carry instead: plan_materializer writes model.name /
        # hardware.label / kv_management.policy / request_queue_csv into
        # manifest.json, and _labels derives dataset from request_queue_csv.
        # M-37 (2026-09-30): a run whose init record declares no
        # manifest_path has no provenance at all -- it must not sit in a
        # label-keyed group with empty label strings (that is the group
        # collapse shape); it gets its own (repo_variant, run_mode) group
        # and an explicit note instead.
        labeled: dict[tuple[Any, ...], list[Run]] = {}
        unlabeled: list[Run] = []
        for run in runs:
            if not run.init.get("manifest_path"):
                unlabeled.append(run)
                continue
            _, service_manifest = _load_manifest_sidecars(run)
            labels = _labels(run, service_manifest, {})
            key = (
                run.init.get("repo_variant", ""),
                run.init.get("run_mode", ""),
                labels["model"],
                labels["dataset"],
                labels["hardware"],
                labels["kv_policy"],
            )
            labeled.setdefault(key, []).append(run)
        member_sets: list[list[Run]] = sorted(
            labeled.values(), key=lambda members: members[0].run_id
        )
        if unlabeled:
            member_sets.append(sorted(unlabeled, key=lambda run: run.run_id))
            if notes is not None:
                notes.append(
                    "auto grouping: "
                    f"{len(unlabeled)} run(s) declare no manifest_path and "
                    "carry no workload labels; grouped separately by init "
                    "identity (labels unavailable, not silently empty)"
                )
        for index, members in enumerate(member_sets):
            group_id = f"auto_{index}"
            _check_group_digest_gate(group_id, members, notes)
            groups.append(
                Group(
                    group_id=group_id,
                    runs=members,
                    baseline_run_id=None,
                    labels_by_run={},
                )
            )
        return groups, run_configs

    # P2 (2026-09-29): a non-object group config used to escape as a bare
    # AttributeError on .get(); structured failure instead (main() checks
    # its own load too -- this guards direct callers).
    if not isinstance(group_config, dict):
        raise PostprocessError(
            f"group config must be a JSON object, got "
            f"{type(group_config).__name__} (sec.11.4)"
        )
    configured_groups = group_config.get("groups")
    if not isinstance(configured_groups, list) or not configured_groups:
        raise PostprocessError("group config must contain a non-empty 'groups' list")
    seen_group_ids: set[str] = set()
    for entry in configured_groups:
        if not isinstance(entry, dict):
            raise PostprocessError(
                f"group config entries must be objects, got "
                f"{type(entry).__name__} (sec.11.4)"
            )
        group_id = entry.get("group_id")
        # M-minor (2026-09-30): the id must be a non-empty *string* -- an
        # unhashable value (list/dict) used to escape as a bare TypeError
        # from the seen-group-ids membership test.
        if not isinstance(group_id, str) or not group_id:
            raise PostprocessError(
                f"group config entry group_id must be a non-empty string, "
                f"got {group_id!r} (sec.11.4)"
            )
        if group_id in seen_group_ids:
            # M24 (2026-09-29): a repeated group_id used to produce two
            # identically named groups (double-counted normalization rows).
            raise PostprocessError(
                f"group config declares group_id {group_id!r} more than "
                "once (sec.11.4); duplicate groups are rejected rather "
                "than silently merged"
            )
        seen_group_ids.add(group_id)
        members: list[Run] = []
        labels_by_run: dict[str, dict[str, Any]] = {}
        baseline_run_id = entry.get("baseline_run_id")
        baseline_marks = 0
        group_run_ids: set[str] = set()
        group_runs_field = entry.get("runs", [])
        # M-minor (2026-09-30): a non-list runs field (or non-object member
        # entries) used to escape as a bare TypeError/AttributeError from
        # the iteration and the first .get.
        if not isinstance(group_runs_field, list) or not group_runs_field:
            raise PostprocessError(
                f"group {group_id}: 'runs' must be a non-empty list "
                "(sec.11.4)"
            )
        for run_entry in group_runs_field:
            if not isinstance(run_entry, dict):
                raise PostprocessError(
                    f"group {group_id}: member entries must be objects, "
                    f"got {type(run_entry).__name__} (sec.11.4)"
                )
            run_id = run_entry.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                # Unhashable/missing member ids used to escape as a bare
                # TypeError from the by-id lookup / membership tests.
                raise PostprocessError(
                    f"group {group_id}: member run_id must be a non-empty "
                    f"string, got {run_id!r} (sec.11.4)"
                )
            run = by_id.get(run_id)
            if run is None:
                raise PostprocessError(
                    f"group {group_id}: run_id {run_id!r} not found in the input logs"
                )
            if run_id in group_run_ids:
                # P2 (2026-09-29): a run repeated inside one group's
                # members list double-counted every denominator.
                raise PostprocessError(
                    f"group {group_id}: run_id {run_id!r} appears more "
                    "than once in the group's members list (sec.11.4)"
                )
            group_run_ids.add(run_id)
            members.append(run)
            labels = {
                key: run_entry[key]
                for key in ("model", "dataset", "hardware", "kv_policy")
                if key in run_entry
            }
            labels_by_run[run_id] = labels
            previous_labels = run_configs.get(run_id)
            if previous_labels is not None and previous_labels != labels:
                # M25 (2026-09-29): a run shared across groups used to get
                # its labels silently overwritten by group order; only a
                # consistent label set is accepted.
                raise PostprocessError(
                    f"group {group_id}: run_id {run_id!r} is a member of "
                    "more than one group with conflicting label overrides "
                    f"{previous_labels!r} vs {labels!r} (sec.11.4)"
                )
            run_configs[run_id] = labels
            if run_entry.get("baseline"):
                baseline_marks += 1
                if baseline_run_id is not None and baseline_run_id != run_id:
                    # P2 (2026-09-29): a baseline_run_id declaration and a
                    # baseline mark (or two marks) pointing at different
                    # runs used to be silently resolved by the mark.
                    raise PostprocessError(
                        f"group {group_id}: conflicting baseline "
                        f"declarations: baseline_run_id "
                        f"{baseline_run_id!r} vs baseline mark on run "
                        f"{run_id!r} (sec.11.4)"
                    )
                baseline_run_id = run_id
        if baseline_marks > 1:
            raise PostprocessError(
                f"group {group_id}: baseline is not unique (sec.11.4)"
            )
        if not entry.get("allow_digest_mismatch", False):
            # M-37 (2026-09-30): the gate moved into the shared helper --
            # an all-empty digest set records an explicit SKIP note instead
            # of a silent vacuous pass, a real mismatch still fails.
            _check_group_digest_gate(group_id, members, notes)
        schemas = set()
        for run in members:
            schemas |= run.schema_versions
        if not _schema_versions_exact(schemas):
            raise PostprocessError(
                f"group {group_id}: mixed metric schema versions "
                f"{sorted(schemas)} must not be normalized together (sec.11.4)"
            )
        groups.append(
            Group(
                group_id=group_id,
                runs=members,
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
                for record in run.iterations:
                    iteration_time = record.get("iteration_time_ns")
                    # M23 (2026-09-29): ``if iteration_time:`` silently
                    # dropped 0/null/missing iterations from the
                    # normalization set while raw_metrics.csv kept the rows
                    # and the run exited 0 -- the service path below fails
                    # closed instead.  A null/missing value is a structured
                    # failure here; 0 reaches the denominator check.
                    if iteration_time is None:
                        raise PostprocessError(
                            f"run {run.run_id}: iteration record "
                            f"(benchmark_point_id="
                            f"{record.get('benchmark_point_id', '')!r}) "
                            "lacks iteration_time_ns; cannot normalize "
                            "(sec.11.4)"
                        )
                    try:
                        iteration_value = float(iteration_time)
                    except (TypeError, ValueError):
                        raise PostprocessError(
                            f"run {run.run_id}: iteration_time_ns "
                            f"{iteration_time!r} is not a number "
                            "(sec.11.4)"
                        ) from None
                    microbench.append((run, record, iteration_value))
            else:
                summary = run.summary
                mean_e2e = summary.get("mean_e2e_ns")
                drain_tput = summary.get("drain_tput_rps")
                if mean_e2e is None or drain_tput is None:
                    raise PostprocessError(
                        f"run {run.run_id}: summary lacks mean_e2e_ns/"
                        "drain_tput_rps; cannot normalize (sec.11.4)"
                    )
                try:
                    # Guard mirrors the microbenchmark branch above: a
                    # non-numeric summary value used to escape as a bare
                    # ValueError from float().
                    mean_value = float(mean_e2e)
                    tput_value = float(drain_tput)
                except (TypeError, ValueError):
                    raise PostprocessError(
                        f"run {run.run_id}: summary mean_e2e_ns/"
                        f"drain_tput_rps ({mean_e2e!r}/{drain_tput!r}) is "
                        "not numeric; cannot normalize (sec.11.4)"
                    ) from None
                service.append((run, mean_value, tput_value))

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

        # P2 (2026-09-29): in group_max mode the denominator is the group
        # max and no baseline run is consumed -- writing the configured
        # baseline run_id into the column implied a baseline denominator
        # that does not exist, so group_max rows carry an empty cell.
        column_baseline_run_id = (
            baseline_run_id or ""
            if method == NORMALIZATION_RATIO_TO_BASELINE
            else ""
        )

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
                    "baseline_run_id": column_baseline_run_id,
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
                    "baseline_run_id": column_baseline_run_id,
                    "denominator_time": "",
                    "denominator_tput": "",
                    "denominator_iteration_time": iteration_den,
                    "normalized_time": "",
                    "normalized_tput": "",
                    "normalized_iteration_time": iteration_record.normalized_value,
                }
            )
    return rows


def _stage_csv(
    path: Path, columns: Sequence[str], rows: Sequence[dict[str, Any]]
) -> Path:
    """Write one complete CSV product to a unique ``.tmp.<pid>.<uuid>``
    staging file next to its final location and return the temp path.

    (M21, 2026-09-29) Callers commit every staged file only after ALL
    products have staged, so a failure can no longer leave half-written
    products mixed with stale run products (request_metrics.csv used to
    fail after raw/normalized had already landed).
    (M-38, 2026-09-30) The staging name is unique per process and per call
    -- the fixed ``<name>.csv.tmp`` used to be one shared slot, so two
    concurrent postprocess runs staging into the same directory stomped
    each other's staged bytes and committed a format-valid, content-mixed
    hybrid CSV; the commit itself stays ``os.replace`` (atomic on POSIX).
    """
    tmp_path = path.with_name(
        f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        with tmp_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(columns))
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in columns})
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return tmp_path


def _commit_staged(tmp_path: Path, path: Path) -> None:
    """Atomically move one staged CSV into place (same directory, same
    filesystem -- ``os.replace`` is atomic on POSIX)."""
    os.replace(tmp_path, path)


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    """Stage one CSV product and commit it atomically."""
    _commit_staged(_stage_csv(path, columns, rows), path)


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

    # sync-py-E4 (2026-09-30): 字面 cpp.log 缺失时透明回退 cpp.log.gz
    # (runner 归档形态;见 _resolve_log_path)。
    log_paths = [_resolve_log_path(Path(log)) for log in args.logs]
    for log_path in log_paths:
        if not log_path.is_file():
            print(f"postprocess error: log not found: {log_path}", file=sys.stderr)
            return 1
    notes: list[str] = []
    group_config = None
    if args.group_config is not None:
        try:
            # E4（2026-09-30）：group config 同走 parse_constant + 有限值
            # 门（NaN 标签会静默分组）。
            group_config = _parse_json_object(
                Path(args.group_config).read_text(encoding="utf-8"),
                "group config " + str(args.group_config))
        except (OSError, PostprocessError) as error:
            print(f"postprocess error: cannot read group config: {error}", file=sys.stderr)
            return 1
        if not isinstance(group_config, dict):
            # P2 (2026-09-29): a non-object group config used to escape as
            # a bare AttributeError inside build_groups.
            print(
                "postprocess error: group config must be a JSON object, "
                f"got {type(group_config).__name__}",
                file=sys.stderr,
            )
            return 1

    try:
        runs = _parse_logs(log_paths)
        if not runs:
            raise PostprocessError("no [METRIC] runs found in the input logs")
        _check_consistency(runs)
        _check_die_utilization_records(runs)
        groups, run_configs = build_groups(runs, group_config, notes)
        raw_rows = build_raw_rows(runs, run_configs, notes)
        normalized_rows = build_normalized_rows(groups, args.normalization, args.baseline)
        # M21 (2026-09-29): the request product is built (all join gates
        # evaluated) BEFORE anything is written, so a join failure can no
        # longer leave raw/normalized frozen next to a stale
        # request_metrics.csv.
        request_rows, request_summary = build_request_metrics(runs)
    except PostprocessError as error:
        print(f"postprocess error: {error}", file=sys.stderr)
        return 1

    # M21: stage all three products to their .tmp files first; only after
    # every product has been materialised completely are the files moved
    # into place (atomic rename).  A staging failure leaves the previous
    # products untouched and no temp files behind.
    out_raw = Path(args.out_raw)
    out_normalized = Path(args.out_normalized)
    out_requests = Path(args.out_requests)
    staged: list[tuple[Path, Path]] = []
    try:
        staged.append(
            (_stage_csv(out_raw, RAW_COLUMNS, raw_rows), out_raw)
        )
        staged.append(
            (_stage_csv(out_normalized, NORMALIZED_COLUMNS, normalized_rows),
             out_normalized)
        )
        if request_rows is not None:
            staged.append(
                (_stage_csv(out_requests, REQUEST_METRICS_COLUMNS,
                            request_rows), out_requests)
            )
    except OSError as error:
        for tmp_path, _ in staged:
            tmp_path.unlink(missing_ok=True)
        print(f"postprocess error: cannot write products: {error}",
              file=sys.stderr)
        return 1
    for tmp_path, final_path in staged:
        _commit_staged(tmp_path, final_path)
    print(json.dumps({"out_requests": str(out_requests), **request_summary}))
    print(
        json.dumps(
            {
                "runs": len(runs),
                "groups": len(groups),
                "raw_rows": len(raw_rows),
                "normalized_rows": len(normalized_rows),
                "out_raw": args.out_raw,
                "out_normalized": args.out_normalized,
                "notes": notes,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
