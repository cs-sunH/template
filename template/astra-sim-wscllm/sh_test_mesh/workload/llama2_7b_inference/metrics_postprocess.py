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
    check_schema_version,
)
from metrics_schema import SCHEMA_VERSION  # noqa: E402

METRIC_PREFIX = "[METRIC] "
# (2026-10-01, R5-F2 discipline) Single authoritative source: the schema
# version constant is imported from metrics_schema (the old local hardcode
# drifted silently on a version bump and wrote a stale label into the
# product at line ~1218). The consumer-side gate now uses the schema
# module's own checker as well.
SUPPORTED_SCHEMA = SCHEMA_VERSION

SOURCE_SIMULATOR = "simulator"
SOURCE_MICROBENCHMARK = "simulator_microbenchmark"
SOURCE_PLANNER_MEMORY = "planner_memory_ledger"

# kv_policy label 取值集（A.5/2026-09-05：legacy 与 relevant 两个历史
# 变体已清除，唯一合法 label session_lru_recompute）。
# 与 generate_wsc_llm_trace._parse_config_value 的 kv_cache_policy 值域同源
# （label 与配置键同名同域）；空串 = 未标注（既有产物/合成 manifest 无
# kv_management 节时缺省），保持向后兼容。非空且不在集合内 → fail-closed
# （防止拼错的 label 静默分裂对比组，sec.11.4 同族防线）。
KV_POLICY_LABELS = frozenset({"session_lru_recompute"})

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


def _schema_versions_exact(schemas: set) -> bool:
    """Type-strict ``schemas == {SUPPORTED_SCHEMA}`` equality.

    F2_PY fix (2026-09-29, RV5-F4 / D2 SC-1 same family): Python set
    equality swallows bool/float masquerades (``{True} == {1}``,
    ``{1.0} == {1}``), so a producer lying with ``"schema": true/1.0``
    walked both the run-level and group-level gates.  The schema version
    must be the literal integer.
    """
    if len(schemas) != 1:
        return False
    value = next(iter(schemas))
    # (2026-10-01) Route the value through metrics_schema's own checker
    # (single authority: _require_int + SCHEMA_VERSION equality) instead of
    # re-implementing the literal-integer rule here; a failure degrades to
    # the same False -> PostprocessError with this file's run context.
    try:
        check_schema_version(value)
    except MetricsSchemaError:
        return False
    return True


def _reject_json_constant(token: str) -> Any:
    """F2_PY fix (2026-09-29, RV5-F10): reject NaN/Infinity/-Infinity JSON
    literals at parse time.

    stdlib json parses these literals to non-finite floats, which the
    summary display columns (p50/p95/p99 etc.) used to carry verbatim into
    raw_metrics.csv with exit code 0 (the A1 P1-4 finite gates only cover
    the judgment fields).  Same fail-closed caliber as
    slo_common.load_json_strict (this module deliberately does not import
    slo_common -- stdlib independent); C++ nlohmann producers serialize
    non-finite floats as null, never as these literals.
    """
    raise PostprocessError(
        f"non-finite JSON literal {token} (NaN/Infinity) in metrics input "
        "(fail-closed; sec.11.4)")


def _resolve_log_path(log_path: Path) -> Path:
    """sync-py-E4 (2026-10-07, ported from the wscllm-LRU repo): 官方 runner 归档
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
                try:
                    record = json.loads(payload,
                                        parse_constant=_reject_json_constant)
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
                # consistency, memory_anchor and unknown types are tolerated
                # but unused.
    seen: dict[str, Run] = {}
    deduped: list[Run] = []
    for run in runs:
        previous = seen.get(run.run_id)
        if previous is not None:
            # A repeated run_id with an identical summary AND identical
            # per-request records is the same run seen twice (e.g. the same
            # log passed twice) -- keep one Run so raw/normalized rows do
            # not double.  Conflicting summaries still fail (sec.11.4).
            # Report 4.3 fix (2026-09-29): the dedup used to compare the
            # summary only, so a second run with a byte-identical summary
            # but diverging request/rank/iteration records was silently
            # dropped record-by-record; the per-record collections are now
            # compared as well and any divergence fails closed.
            if previous.summary != run.summary:
                raise PostprocessError(
                    f"run_id {run.run_id!r} appears in multiple logs with "
                    "conflicting summaries (sec.11.4)"
                )
            if (previous.requests != run.requests
                    or previous.rank_compute != run.rank_compute
                    or previous.iterations != run.iterations
                    or previous.capacity_timeavg != run.capacity_timeavg
                    or previous.planner_peaks != run.planner_peaks):
                raise PostprocessError(
                    f"run_id {run.run_id!r} appears in multiple logs with "
                    "identical summaries but diverging per-request or "
                    "per-measurement records (sec.11.4; no silent record "
                    "drop -- dedup requires byte-identical record sets)"
                )
            continue
        seen[run.run_id] = run
        deduped.append(run)
    runs = deduped
    for run in runs:
        if not _schema_versions_exact(run.schema_versions):
            raise PostprocessError(
                f"run {run.run_id}: unsupported or mixed metric schema "
                f"versions {sorted(run.schema_versions)} (sec.11.4); this "
                f"postprocessor supports schema {SUPPORTED_SCHEMA} only"
            )
    return runs


def _load_manifest_sidecars(run: Run) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (metrics_manifest, service manifest.json)."""

    metrics_manifest: dict[str, Any] = {}
    service_manifest: dict[str, Any] = {}
    manifest_path = run.init.get("manifest_path")
    if not manifest_path:
        return metrics_manifest, service_manifest
    path = Path(manifest_path)
    try:
        metrics_manifest = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant)
    except (OSError, json.JSONDecodeError) as error:
        # 读取/解析失败降级为空 manifest 时必须留痕：否则 request 路径
        # 最终以误导性的 no-request-entries fail-closed，真实原因被掩盖。
        print(
            f"postprocess warning: cannot read {path} ({error}); "
            "continuing without it",
            file=sys.stderr,
        )
        metrics_manifest = {}
    try:
        service_manifest = json.loads(
            (path.parent / "manifest.json").read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant
        )
    except (OSError, json.JSONDecodeError) as error:
        print(
            f"postprocess warning: cannot read {path.parent / 'manifest.json'} "
            f"({error}); labels degrade to empty strings",
            file=sys.stderr,
        )
        service_manifest = {}
    return metrics_manifest, service_manifest


# die 后端利用率三元组的管线级断言(wscllm die 推广)。C++ MetricCollector
# 在每条 type=rank_compute 记录上输出 active_kernel_die_{compute,memory,
# vector}_util(门控=compute-cost-model=die;roofline 键保持 legacy 门控
# 不动)。本断言在后处理侧 fail-closed:声明 die 后端的 run,每个 rank 的
# 三个键必须非 null、数值、且落在 [0, 1+eps](eps 与 C++ kUtilEpsilon 同值)。
DIE_UTILIZATION_KEYS = (
    "active_kernel_die_compute_util",
    "active_kernel_die_memory_util",
    "active_kernel_die_vector_util",
)
# 与 C++ MetricCollector 的 kUtilEpsilon 一致(上行在 1e-6 容差内钳制)。
DIE_UTILIZATION_EPSILON = 1e-6
# null 的唯一合法理由:窗口内没有任何 comp 节点(C++ 端 note 原文)。
_DIE_NULL_NOTE_OK = "unavailable: no comp nodes in the utilization window"


def _run_declares_die_backend(run: Run) -> bool:
    """该 run 是否声明 die 计算后端。

    声明源优先级:metrics_manifest.json 的 compute-cost-model(plan_
    materializer 同步写入,归档 run 目录自带)> 同目录 manifest.json 的
    同名键(服务 manifest)> 无声明时从 rank_compute 记录自身推断(C++
    的 note 文案区分 not-die)。
    """

    metrics_manifest, service_manifest = _load_manifest_sidecars(run)
    # 归档 run 目录兜底:init 的 manifest_path 不在(或指向别处)时,按
    # 日志同目录的 metrics_manifest.json / manifest.json 发现(runner 把
    # 两份 manifest 都拷进 run 目录;读取失败静默跳过,由记录自推断收尾)。
    sidecar_candidates = (
        run.log_path.parent / "metrics_manifest.json",
        run.log_path.parent / "manifest.json",
    )
    manifests = [metrics_manifest, service_manifest]
    for sidecar in sidecar_candidates:
        try:
            manifests.append(json.loads(
                sidecar.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant))
        except (OSError, json.JSONDecodeError):
            continue
    for manifest in manifests:
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
    """Pipeline assertion: die-backend utilization triplet.

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


def _check_request_completion(run: Run) -> None:
    """sec.11.4: an incomplete request must never be silently counted."""

    summary = run.summary
    if not run.requests:
        return  # summary detail level: nothing to cross-check against
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


def _int_or_na(value: Any, field: str = "") -> Any:
    """Render a metrics JSON integer, or the frozen NA placeholder.

    D2 fix (2026-09-28): a non-numeric non-null value used to escape as a
    bare int() ValueError/TypeError (bypassing the PostprocessError
    channel, so main() died with a traceback instead of the sec.11.4
    "postprocess error" exit); it now aborts through the same channel.
    """
    if value is None:
        return NA
    try:
        return int(value)
    except (TypeError, ValueError):
        raise PostprocessError(
            f"request record field {field or '?'} is not an integer or "
            f"null (got {value!r}; fail-closed)"
        ) from None


def _int_or_none(value: Any) -> Optional[int]:
    """Metrics JSON integer or None (WP9 proxy arithmetic helper)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _manifest_request_entries(
    run: Run,
    metrics_manifest: dict[str, Any],
    service_manifest: dict[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    """Per-request manifest entries keyed by queue_index (fail-closed join).

    Merges the run's ``metrics_manifest.json`` (the C++ collector input)
    with its sibling service ``manifest.json``.  Both must agree on
    ``request_id`` whenever they carry it for the same ``queue_index``;
    fields absent from one manifest are taken from the other.  Returns
    ``(entries_by_queue_index, per_manifest_request_counts)``.
    """
    merged: dict[int, dict[str, Any]] = {}
    # D2 fix (2026-09-28): only manifests that actually DECLARE a requests
    # list participate in the per-manifest count check.  The former
    # `counts[name] = 0` for an absent/non-list requests key contradicted
    # this function's own one-sided tolerance (fields absent from one
    # manifest are taken from the other) -- write_request_metrics then
    # failed every run whose service manifest carries no requests array
    # with a misleading "declares 0 requests" message.  A manifest that
    # declares an empty list still counts as declaring 0 and fails the
    # check when the log carries records.
    counts: dict[str, int] = {}
    for name, manifest in (
        ("metrics_manifest.json", metrics_manifest),
        ("manifest.json", service_manifest),
    ):
        requests = manifest.get("requests")
        if not isinstance(requests, list):
            continue  # not declared: nothing to count (one-sided tolerance)
        counts[name] = len(requests)
        for entry in requests:
            if not isinstance(entry, dict) or "queue_index" not in entry:
                raise PostprocessError(
                    f"run {run.run_id}: {name} request entry without "
                    "queue_index cannot be joined to the request records"
                )
            queue_index = entry["queue_index"]
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
                    if (
                        key in combined
                        and value is not None
                        and combined[key] != value
                        and key != "request_id"
                    ):
                        raise PostprocessError(
                            f"run {run.run_id}: queue_index {queue_index} "
                            f"has conflicting {key} across manifests "
                            f"({value!r} vs {combined[key]!r})"
                        )
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


def _int_field(value: Any, field: str, run_id: str, queue_index: Any) -> int:
    """Required metrics JSON integer (D2 fix: fail through PostprocessError,
    never a bare int() ValueError/TypeError)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        raise PostprocessError(
            f"run {run_id}: queue_index {queue_index!r}: record field "
            f"{field} is not an integer (got {value!r}; fail-closed)"
        ) from None


def _check_request_row_invariants(
    run_id: str, record: dict[str, Any], queue_index: Any
) -> None:
    """Ordering/decomposition invariants on rows that carry values.

    Checked only among the fields that are present: arrival <= first_token
    <= completion (first_token skipped while NA), e2e == completion -
    arrival, and queue+prefill+gap+decode == e2e.  Any violation aborts
    (no silent clamping).  D2 fix (2026-09-28): a present non-numeric
    field used to escape the comparisons as a bare TypeError (same
    PostprocessError-channel bypass as the rendering path) -- every field
    read here must be integer-coercible or None.
    """
    checked_fields = (
        "arrival_ns", "first_token_ns", "completion_ns", "e2e_ns",
        "queue_ns", "prefill_ns", "prefill_decode_gap_ns", "decode_ns",
    )
    record = dict(record)
    for name in checked_fields:
        value = record.get(name)
        if value is None:
            continue
        try:
            record[name] = int(value)
        except (TypeError, ValueError):
            raise PostprocessError(
                f"run {run_id}: queue_index {queue_index}: field {name} "
                f"is not an integer or null (got {value!r}; fail-closed)"
            ) from None
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

# Directories searched for trace_config.csv by the first-token proxy, in
# order, seeded by main() with the run dirs of the logs being postprocessed
# (run-dir frozen copy wins; MODULE_DIR is the permanent fallback).
_CONFIG_SEARCH_DIRS: Optional[list[Path]] = None


def _trace_config_candidates() -> list[Path]:
    dirs: list[Path] = list(_CONFIG_SEARCH_DIRS) if _CONFIG_SEARCH_DIRS else []
    dirs.append(MODULE_DIR)
    return [directory / "trace_config.csv" for directory in dirs]


def _load_model_bytes() -> tuple[Optional[int], Optional[int]]:
    """(W_bytes, kv_bytes_per_token) from this workload's trace_config.csv.

    Candidate order: the run-dir frozen copy the runner stages next to
    cpp.log first, then the repo-current file -- re-running this
    postprocessor over an archived run dir after a campaign changed the
    model rows must keep deriving the proxy from the run's own parameters
    (same run_dir-first discipline as hbm_watermark.load_trace_config).

    Returns (None, None) when no candidate exists or a model row is
    missing/invalid (proxy then degrades to NA; nothing is fabricated).
    """

    try:
        from wsc_llm_scheduler import (
            WscLlmModel,
            estimate_model_weight_bytes,
            kv_cache_bytes_for_tokens,
        )
    except ImportError:
        return None, None
    config_path: Optional[Path] = None
    for candidate in _trace_config_candidates():
        if candidate.is_file():
            config_path = candidate
            break
    if config_path is None:
        return None, None
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
        self._loaded = False
        # request_id -> (ledger_row_position, row) of its first (non
        # first_step) train carrying it in joiners.
        self.first_train_by_request: dict[str, tuple[int, dict[str, Any]]] = {}
        # ledger row position -> tick of the next row on the same instance.
        self.next_tick_by_position: dict[int, int] = {}
        # Why the proxy is unavailable (instructions note): an absent ledger
        # and a corrupt ledger are different failures -- a corrupt line is
        # reported with its line number instead of masquerading as
        # "ledger missing".
        self.unavailable_reason = "no_train_ledger(results/)"

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
            return  # unavailable_reason keeps the absent-ledger note
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
                        self.unavailable_reason = (
                            f"train_ledger_corrupt(line {line_number}: "
                            f"{error.msg})"
                        )
                        return
                    if not isinstance(record, dict):
                        # Same channel as corrupt lines (2026-09-30): a
                        # valid-JSON-but-non-object row is a third damage
                        # shape that used to vanish silently (neither
                        # counted nor reported).
                        self.unavailable_reason = (
                            f"train_ledger_non_object(line {line_number})")
                        return
                    rows.append(record)
        except OSError as error:
            # 与 corrupt 同理：ledger 在场但读不了也不是"不存在"，
            # 行级备注要如实区分（不留缺省的 no_train_ledger 误报）。
            self.unavailable_reason = f"train_ledger_unreadable({error})"
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
            notes.append(f"proxy_unavailable:{proxy_index.unavailable_reason}")
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
        # 4.3 fix (2026-09-29): the gate compared the RAW entry value
        # (`== 1`), so a JSON string "1" silently bypassed the dl1
        # invariant while the row's own length lexicon (_int_or_none)
        # accepted it. Parse through the same lens first (口径一致).
        dl1_decode_length = _int_or_none(entry.get("decode_length"))
        if (first_token_source == FIRST_TOKEN_SOURCE_TRAIN_INTERPOLATED
                and first_token_ns is not None
                and completion_ns is not None
                and dl1_decode_length == 1
                and first_token_ns != completion_ns):
            raise PostprocessError(
                f"run {run.run_id}: queue_index {queue_index} request "
                f"{record.get('request_id')!r}: decode_length==1 proxy "
                "first_token_ns != completion_ns (dl1 first token IS the "
                "last token; pinned_to_completion path broken)"
            )
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
        rows.append(
            {
                "queue_index": _int_field(
                    queue_index, "queue_index", run.run_id, queue_index),
                "request_id": record.get("request_id", ""),
                "session_id": record.get("session_id", ""),
                "turn_index": _int_field(
                    record.get("turn_index", 0), "turn_index",
                    run.run_id, queue_index),
                "request_type": request_type,
                "terminal_status": terminal_status,
                "arrival_ns": _int_or_na(record.get("arrival_ns"), "arrival_ns"),
                "prefill_start_ns": _int_or_na(
                    record.get("prefill_start_ns"), "prefill_start_ns"),
                "prefill_end_ns": _int_or_na(
                    record.get("prefill_end_ns"), "prefill_end_ns"),
                "decode_start_ns": _int_or_na(
                    record.get("decode_start_ns"), "decode_start_ns"),
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
                    record.get("completion_ns"), "completion_ns"),
                "queue_ns": _int_or_na(record.get("queue_ns"), "queue_ns"),
                "prefill_ns": _int_or_na(record.get("prefill_ns"), "prefill_ns"),
                "prefill_decode_gap_ns": _int_or_na(
                    record.get("prefill_decode_gap_ns"),
                    "prefill_decode_gap_ns"
                ),
                "decode_ns": _int_or_na(record.get("decode_ns"), "decode_ns"),
                "e2e_ns": _int_or_na(record.get("e2e_ns"), "e2e_ns"),
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
    # F2_PY fix (2026-09-29, RV5-F3): a well-formed-JSON but malformed-shape
    # manifest.json (e.g. "model": "llama2" -- a string, not an object) used
    # to escape the chained .get as a bare AttributeError.  Report 3.6
    # follow-up (2026-09-29): the original guard exempted explicit JSON null
    # blocks ("model": null), which then still crashed the same way on the
    # chained .get below; a null block is a shape violation too, so only a
    # dict block passes.  The label blocks are comparison-group keys (the
    # kv_policy value below is already fail-closed on unknown values), so a
    # wrong shape aborts through PostprocessError instead of escaping as a
    # bare traceback.
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
    if kv_policy and str(kv_policy) not in KV_POLICY_LABELS:
        # A.5/2026-09-05：kv_policy label 枚举校验（唯一合法值
        # session_lru_recompute）；空串=未标注仍放行（既有产物兼容），
        # 非空未知值 fail-closed。
        raise PostprocessError(
            f"kv_policy label {kv_policy!r} not in the known set "
            f"{sorted(KV_POLICY_LABELS)} (typo would silently split "
            "comparison groups)")
    dataset = run_config.get("dataset") or ""
    if not dataset:
        queue_csv = service_manifest.get("request_queue_csv", "")
        if not isinstance(queue_csv, str):
            # RV7-1 (2026-09-29): same family as the three blocks above -- a
            # non-string queue path (hand-edited manifest) used to escape as
            # a bare TypeError from Path().  Report 3.6 follow-up (same
            # date): an explicit JSON null slipped past the original
            # `is not None` exemption and silently degraded the label to
            # ""; a null path is a shape violation too and now takes the
            # same PostprocessError channel.
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
    run: Run, labels: dict[str, str], metrics_manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    point = metrics_manifest.get("microbenchmark", {})
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
    active_ranks = set(point.get("ranks", []))
    ops_by_rank: dict[int, int] = {}
    bytes_by_rank: dict[int, int] = {}
    peak_flops: Optional[float] = None
    local_bw: Optional[float] = None

    def _rank_int(record: dict[str, Any], field: str) -> int:
        # F2_PY fix (2026-09-29, RV5-F2): a non-numeric total_num_ops /
        # total_local_bytes used to escape as a bare int() ValueError/
        # TypeError (D2 MP-3 collection-point miss) -- same
        # PostprocessError channel as _int_field, message carries the
        # run id and field name.
        try:
            return int(record.get(field, 0))
        except (TypeError, ValueError):
            raise PostprocessError(
                f"run {run.run_id}: rank_compute record field {field} is "
                f"not an integer (got {record.get(field)!r}; fail-closed)"
            ) from None

    _reject_duplicate_rank_records(run, run.rank_compute, "rank_compute")
    for record in run.rank_compute:
        rank = record.get("rank")
        if active_ranks and rank not in active_ranks:
            continue
        ops_by_rank[rank] = _rank_int(record, "total_num_ops")
        bytes_by_rank[rank] = _rank_int(record, "total_local_bytes")
        # First active rank wins (the C++ rank_compute denominators come
        # from that rank's own system configuration at finalize time);
        # a later rank must not silently replace it on heterogeneous
        # per-rank hardware.  (D2 fix 2026-09-28: the former comment
        # cited "reference_sys in MetricCollector.cc" -- no such entity
        # exists in this repository.)
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
            # F2_PY fix (2026-09-29, RV5-F2): a truthy non-numeric operand
            # (e.g. iteration_time_ns "1000") escaped the division as a
            # bare TypeError (same D2 MP-3 channel bypass) -- abort through
            # PostprocessError with the run id; the arithmetic itself is
            # unchanged for numeric operands.
            try:
                window_s = iteration_time_ns / 1e9
                # doc sec.8.7, recomputed from the rank_compute records.  The
                # C++ finalize emits type=rank_compute records that carry
                # achieved_compute_util under the same formula
                # (num_ops / (peak_flops_per_second * window_seconds)), gated
                # on the roofline (nullptr otherwise -- MetricCollector.cc
                # finalize); this recomputation is the roofline-independent
                # port of that contract.  (D2 fix 2026-09-28: the former
                # comment referenced "reference_sys" and a C++ "iteration
                # record" -- neither entity exists in this repository; there
                # is no iteration-record producer here either, so this whole
                # per-iteration branch is contract-latent, see P2-22.)
                # Registered limitation (D2 2026-09-28, contract-latent):
                # rank_compute totals are per RUN while the denominator below
                # is per ITERATION -- with N iterations per point each row's
                # ratio would inflate ~N-fold.  Untouchable without a producer
                # that attributes ops/bytes per iteration; this repo has none.
                compute_util = total_ops / (tp_degree * peak_flops * window_s)
                hbm_bw_util = total_bytes / (tp_degree * local_bw * window_s)
            except (TypeError, ValueError):
                raise PostprocessError(
                    f"run {run.run_id}: microbench iteration utilization "
                    "operands are not all numeric "
                    f"(iteration_time_ns={iteration_time_ns!r}, "
                    f"tp_degree={tp_degree!r}, "
                    f"peak_flops_per_second={peak_flops!r}, "
                    f"local_hbm_bw_bytes_per_second={local_bw!r}; "
                    "fail-closed)"
                ) from None
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


def _reject_duplicate_rank_records(
        run: "Run", records: list, record_type: str) -> None:
    """Rank-cardinality reconciliation (2026-09-30): the C++ writer promises
    exactly one <record_type> record per rank per run; a duplicate means a
    log-assembly defect (double append / spliced logs) and previously
    silently overwrote (planner_peaks / rank_compute) or double-counted
    (capacity_timeavg raw rows). Fail closed like the same-run summary
    conflict gate instead of emitting distorted rows."""
    seen: set = set()
    for record in records:
        rank = record.get("rank")
        if rank in seen:
            raise PostprocessError(
                f"run {run.run_id}: duplicate {record_type} record for rank "
                f"{rank!r} (the writer emits one record per rank; a "
                "duplicate would silently distort the per-rank rows)"
            )
        seen.add(rank)


def _memory_ledger_rows(run: Run, labels: dict[str, str]) -> list[dict[str, Any]]:
    _reject_duplicate_rank_records(run, run.planner_peaks,
                                   "planner_memory_peaks")
    _reject_duplicate_rank_records(run, run.capacity_timeavg,
                                   "capacity_timeavg")
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


def build_raw_rows(
    runs: Sequence[Run], run_configs: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        _check_request_completion(run)
        metrics_manifest, service_manifest = _load_manifest_sidecars(run)
        labels = _labels(run, service_manifest, run_configs.get(run.run_id, {}))
        run_mode = run.init.get("run_mode", "service")
        if run_mode == "microbenchmark":
            rows.extend(_microbench_rows(run, labels, metrics_manifest))
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
            # P2-22 fix (2026-09-28): microbenchmark runs carry their measured
            # benchmark_point_id set into the auto key -- runs measuring
            # different point sets must never share a normalization group
            # (cross-point denominators are meaningless; sec.11.4).  Service
            # runs keep the legacy key (empty point set).
            if run.init.get("run_mode") == "microbenchmark":
                point_ids = tuple(sorted({
                    str(record.get("benchmark_point_id", ""))
                    for record in getattr(run, "iterations", [])
                }))
            else:
                point_ids = ()
            key = (
                run.init.get("repo_variant", ""),
                run.init.get("run_mode", ""),
                run.init.get("trace_digest", ""),
                run.init.get("request_mapping_digest", ""),
                point_ids,
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
                    labels_by_run={},
                )
            )
        return groups, run_configs

    # Report 3.6 fix (2026-09-29): a --group-config JSON value of the wrong
    # shape (a top-level array/string/number, a non-object group entry, a
    # non-object run entry) used to escape the chained .get calls below as a
    # bare AttributeError -- outside the PostprocessError channel that main()
    # is the only catcher of.  The group config drives comparison-group keys,
    # so every level is validated as the object shape its access pattern
    # requires and a wrong shape aborts fail-closed (sec.11.4 family).
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
        member_ids: set[str] = set()
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
            # D2 fix (2026-09-28): a duplicated run_id inside one group used
            # to be appended twice, silently double-counting the run in
            # every group statistic and emitting two identical normalized
            # rows (sec.11.4 family: no silent double counting).
            if run_id in member_ids:
                raise PostprocessError(
                    f"group {group_id}: run_id {run_id!r} is listed more "
                    "than once in the group (sec.11.4; no silent double "
                    "counting)"
                )
            member_ids.add(run_id)
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

        # Service rows: one per run.  Microbenchmark rows: one per
        # iteration -- a benchmark point spans multiple iteration records,
        # so one point yields many rows and the normalized value is the
        # per-iteration value.  An iteration without a usable
        # iteration_time_ns still emits a row (normalized_iteration_time
        # carries the NA placeholder), so the normalized CSV stays aligned
        # with raw_metrics.csv instead of silently skipping rows.
        service: list[tuple[Run, float, float]] = []
        microbench: list[tuple[Run, dict[str, Any], float]] = []
        null_iterations: list[tuple[Run, dict[str, Any]]] = []
        def _finite(value: Any, name: str, run: Run) -> float:
            # D2 fix (2026-09-28): a non-numeric value (e.g. the string
            # "NA") escaped float() as a bare ValueError/TypeError outside
            # the PostprocessError channel -- reject it with the run id,
            # same collection-point discipline as the non-finite gates.
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise PostprocessError(
                    f"run {run.run_id}: {name} is not a number "
                    f"({value!r}); cannot normalize (sec.11.4)"
                ) from None
            if not math.isfinite(number):
                # P1 fix (2026-09-28): same non-finite hole as
                # the service summaries -- reject at collection.
                raise PostprocessError(
                    f"run {run.run_id}: {name} is not a finite number "
                    f"({value!r}); cannot normalize (sec.11.4)"
                )
            return number

        for run in group.runs:
            if run.init.get("run_mode") == "microbenchmark":
                for record in run.iterations:
                    iteration_time = record.get("iteration_time_ns")
                    if iteration_time:
                        microbench.append(
                            (run, record,
                             _finite(iteration_time, "iteration_time_ns", run)))
                    else:
                        null_iterations.append((run, record))
            else:
                summary = run.summary
                mean_e2e = summary.get("mean_e2e_ns")
                drain_tput = summary.get("drain_tput_rps")
                if mean_e2e is None or drain_tput is None:
                    raise PostprocessError(
                        f"run {run.run_id}: summary lacks mean_e2e_ns/"
                        "drain_tput_rps; cannot normalize (sec.11.4)"
                    )
                # P1 fix (2026-09-28): a NaN/Inf summary value would slip
                # every "is None"/"<= 0" gate below (nan <= 0 is False) and
                # poison the whole group's normalized columns with nan while
                # main() still exits 0 -- reject at collection with the run
                # id in the message.
                service.append(
                    (run,
                     _finite(mean_e2e, "mean_e2e_ns", run),
                     _finite(drain_tput, "drain_tput_rps", run)))

        def denominator(values: list[float], baseline_value: Optional[float]) -> float:
            if method == NORMALIZATION_GROUP_MAX:
                if not values:
                    raise PostprocessError(
                        f"group {group.group_id}: empty normalization group "
                        "(sec.11.4)"
                    )
                # P1 fix (2026-09-28): max() propagates a NaN sitting in the
                # FIRST slot and silently drops one elsewhere -- every group
                # value must be finite before the max is trusted.
                for item in values:
                    if not math.isfinite(item):
                        raise PostprocessError(
                            f"group {group.group_id}: normalization group "
                            f"value is not finite ({item!r}) (sec.11.4)"
                        )
                value = max(values)
            else:
                if baseline_value is None:
                    raise PostprocessError(
                        f"group {group.group_id}: baseline run has no usable "
                        "value for normalization (sec.11.4)"
                    )
                value = baseline_value
            # P1 fix (2026-09-28): the former `value <= 0` gate let NaN
            # through (nan <= 0 is False); the denominator must be finite
            # AND positive.
            if not (math.isfinite(value) and value > 0):
                raise PostprocessError(
                    f"group {group.group_id}: normalization denominator is "
                    f"not finite or not positive ({value!r}) (sec.11.4)"
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
        # P2-22 fix (2026-09-28): microbench normalization must align by
        # benchmark_point_id.  The legacy path computed ONE denominator for
        # the whole group -- group_max across points, or the baseline run's
        # FIRST iteration regardless of point -- so cross-point ratios were
        # meaningless (a 10x-larger point dominates every smaller point's
        # normalized value).  Records are partitioned by point id; each
        # point gets its own denominator (group max within the point, or
        # the baseline run's value within the SAME point).  Groups whose
        # records carry no benchmark_point_id keep the exact legacy
        # single-bucket behavior: every id is "" so the partition is one
        # bucket over all records (this repo has no iteration producer yet;
        # the point-id branch is contract-latent and pinned by tests).
        micro_by_point: dict[str, list[tuple[Run, dict[str, Any], float]]] = {}
        row_slot: list[tuple[str, int]] = []
        for run, record, value in microbench:
            point_id = str(record.get("benchmark_point_id", ""))
            members = micro_by_point.setdefault(point_id, [])
            row_slot.append((point_id, len(members)))
            members.append((run, record, value))
        point_den: dict[str, Any] = {}
        point_normalized: dict[str, list[Any]] = {}
        for point_id, members in micro_by_point.items():
            baseline_value = (
                next(
                    (value for run, _, value in members
                     if run.run_id == baseline_run_id),
                    None,
                )
                if baseline_run_id
                else None
            )
            point_den[point_id] = denominator(
                [value for _, _, value in members], baseline_value)
            point_normalized[point_id] = [
                _normalize_value(
                    method=method,
                    metric_name="iteration_time_ns",
                    raw_value=value,
                    group_id=group.group_id,
                    group_values=[value for _, _, value in members],
                    baseline_run_id=baseline_run_id,
                    baseline_value=baseline_value,
                )
                for _, _, value in members
            ]

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
        for (run, record, iteration_time), (point_id, slot) in zip(
                microbench, row_slot):
            # P2-22 fix (2026-09-28): per-point denominator and per-point
            # normalized record (computed in the partition pass above).
            iteration_record = point_normalized[point_id][slot]
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
                    "denominator_iteration_time": point_den[point_id],
                    "normalized_time": "",
                    "normalized_tput": "",
                    "normalized_iteration_time": iteration_record.normalized_value,
                }
            )
        for run, record in null_iterations:
            # The NA-normalized counterpart of an iteration record that
            # carries no usable iteration_time_ns (rows are never skipped;
            # see the collection loop above).
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
                    "denominator_iteration_time": "",
                    "normalized_time": "",
                    "normalized_tput": "",
                    "normalized_iteration_time": NA,
                }
            )
    return rows


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


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

    # sync-py-E4 (2026-10-07): 字面 cpp.log 缺失时透明回退 cpp.log.gz
    # (runner 归档形态;见 _resolve_log_path)。
    log_paths = [_resolve_log_path(Path(log)) for log in args.logs]
    # Seed the first-token proxy's config search order with the run dirs of
    # the logs being postprocessed (frozen trace_config.csv copy first,
    # repo-current file as fallback); distinct run dirs keep their own
    # order of appearance.
    seen_dirs: set[Path] = set()
    seeded_dirs: list[Path] = []
    for log_path in log_paths:
        run_dir = log_path.resolve().parent
        if run_dir not in seen_dirs:
            seen_dirs.add(run_dir)
            seeded_dirs.append(run_dir)
    global _CONFIG_SEARCH_DIRS
    _CONFIG_SEARCH_DIRS = seeded_dirs
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
        _check_die_utilization_records(runs)
    except PostprocessError as error:
        print(f"postprocess error: {error}", file=sys.stderr)
        return 1

    _write_csv(Path(args.out_raw), RAW_COLUMNS, raw_rows)
    _write_csv(Path(args.out_normalized), NORMALIZED_COLUMNS, normalized_rows)
    try:
        write_request_metrics(runs, Path(args.out_requests))
    except PostprocessError as error:
        print(f"postprocess error: {error}", file=sys.stderr)
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
