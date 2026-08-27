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
- ``request_metrics.csv`` (WP1, SLO B1, only with ``--out-request``): one row
  per request of a single full-detail service run, joined fail-closed against
  the sibling ``manifest.json`` requests[] (queue_index/request_id).  The
  column set is frozen by the SLO execution plan sec.2; fields not yet
  produced by later work packages are written as ``NA``.  ``off``/``summary``
  detail levels carry no per-request records, so the file is skipped with an
  explanatory line on stdout.  ``first_token_ns``: exact when the WP9 code-8
  events exist; otherwise the spec 1.6 train-interpolated proxy (B4 fallback
  after the B3_S1 60s decision-equivalence gate failure, see the
  ``_TrainProxyIndex`` block) tagged ``first_token_source=train_interpolated``
  -- display metric only, never a SLO judgment input.

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

# WP1 (SLO B1): request_metrics.csv column set -- frozen by the SLO execution
# plan sec.2 ("列集一次定死，后续只填值不加列").  B1 fills the timing /
# terminal_status / instructions block; every later work package only replaces
# NA cells, never adds columns.
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

REQUEST_TYPE_VALUES = frozenset({"human", "tool", "unknown"})
REQUEST_METRICS_NA = "NA"

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


def _parse_logs(log_paths: Sequence[Path]) -> list[Run]:
    runs: list[Run] = []
    for log_path in log_paths:
        current: Optional[Run] = None
        run_index = 0
        with log_path.open(encoding="utf-8", errors="replace") as source:
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
    seen: dict[str, Run] = {}
    for run in runs:
        previous = seen.get(run.run_id)
        if previous is not None:
            if previous.summary != run.summary:
                raise PostprocessError(
                    f"run_id {run.run_id!r} appears in multiple logs with "
                    "conflicting summaries (sec.11.4)"
                )
        seen[run.run_id] = run
    for run in runs:
        if run.schema_versions != {SUPPORTED_SCHEMA}:
            raise PostprocessError(
                f"run {run.run_id}: unsupported or mixed metric schema "
                f"versions {sorted(run.schema_versions)} (sec.11.4); this "
                f"postprocessor supports schema {SUPPORTED_SCHEMA} only"
            )
    return runs


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


def _request_int(record: dict[str, Any], key: str, queue_index: Any) -> Optional[int]:
    value = record.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise PostprocessError(
            f"queue_index {queue_index}: request record field {key} is not an "
            f"integer: {value!r}"
        )
    return value


def _int_or_none(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# WP9 fallback (B4, 2026-08-27): train-interpolated first-token proxy.
#
# B3_S2 60s gate-2 (decision equivalence) failed -- the split's physical
# perturbation (one extra delivery per split train -> tick shifts) is
# amplified by the closed loop (see /tmp/slo_wps/gates/B3_S2.FAILED,
# evidence /tmp/slo_wps/b3/S2/) -- so the default first-token source falls
# back to the spec 1.6 proxy (SH_FIRST_TOKEN_SPLIT now defaults to "0"):
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
# entries == requests in the 60s reference run), hence consumed=0.
#
# first_train_end_ns: the train ledger row's tick is the emission (start)
# boundary; an instance is busy until the train's end barrier completes
# (one-train-in-flight gate, sh20_online_scheduler.py in_flight_train), so
# the NEXT ledger row on the same instance is the first decision boundary
# after the end barrier and is used as the train-end tick (last train of an
# instance -> proxy not obtainable -> NA).  Degenerate 1-iteration trains
# (N=1 -> share=1) make the proxy equal that next-emission boundary, which
# sits after the barrier tick recorded as completion; such values are
# clamped to completion (the first token physically cannot complete after
# the request) and flagged "clamped_to_completion" in instructions.
#
# W_bytes / KV bytes per token: the repository's authoritative conversions
# (face_scheduler.estimate_model_weight_bytes -- weights are read once per
# iteration -- and kv_cache_bytes_for_tokens), parameterised by the
# trace_config.csv model rows; no value is invented here.  sh_2.0 layered
# KV: kv_cache_shard_bytes_for_layer_range partitions the SAME whole-head
# total per layer range (sum(shards) == kv_cache_bytes_for_tokens verified
# at runtime), so the per-token weight above is the authoritative conversion
# here too -- layer ranges only split WHERE bytes live, not the total.
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
        except (OSError, json.JSONDecodeError):
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
    # and overshoot completion.
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
    return proxy_ns, note


def _request_metric_rows(
    run: Run, service_manifest: dict[str, Any]
) -> tuple[Optional[list[dict[str, Any]]], dict[str, Any]]:
    """Build request_metrics.csv rows for one full-detail service run.

    Returns ``(rows, summary)``; ``rows is None`` means the run legitimately
    carries no per-request records (off/summary detail, microbenchmark) and
    ``summary["reason"]`` explains the skip.  Fail-closed (sec.11.4 spirit):
    any manifest<->request-record join failure (unknown/duplicate
    queue_index, request_id mismatch, row-count mismatch) or timing
    invariant violation raises :class:`PostprocessError`.
    """

    detail = run.init.get("detail_level")
    if run.init.get("run_mode") == "microbenchmark":
        return None, {"reason": f"run_mode=microbenchmark carries no per-request records"}
    if not run.requests:
        if detail == "full":
            raise PostprocessError(
                f"run {run.run_id}: detail_level=full but no request records "
                "were found in the log"
            )
        return None, {
            "reason": (
                f"detail_level={detail!r}: per-request records are emitted at "
                "detail level full only"
            )
        }

    manifest_requests = service_manifest.get("requests")
    if not isinstance(manifest_requests, list) or not manifest_requests:
        raise PostprocessError(
            f"run {run.run_id}: manifest.json requests[] is missing or empty; "
            "cannot join per-request metrics (fail-closed)"
        )
    entry_by_queue_index: dict[int, dict[str, Any]] = {}
    for entry in manifest_requests:
        if not isinstance(entry, dict):
            raise PostprocessError(
                f"run {run.run_id}: manifest.json requests[] holds a non-object "
                f"entry: {entry!r}"
            )
        queue_index = entry.get("queue_index")
        if isinstance(queue_index, bool) or not isinstance(queue_index, int):
            raise PostprocessError(
                f"run {run.run_id}: manifest.json request entry lacks integer "
                f"queue_index: {entry!r}"
            )
        if queue_index in entry_by_queue_index:
            raise PostprocessError(
                f"run {run.run_id}: manifest.json requests[] has duplicate "
                f"queue_index {queue_index}"
            )
        entry_by_queue_index[queue_index] = entry
    if len(entry_by_queue_index) != len(run.requests):
        raise PostprocessError(
            f"run {run.run_id}: manifest declares {len(entry_by_queue_index)} "
            f"requests but the log carries {len(run.requests)} request records "
            "(row-count mismatch, fail-closed)"
        )

    def sort_key(record: dict[str, Any]) -> int:
        queue_index = record.get("queue_index")
        if isinstance(queue_index, bool) or not isinstance(queue_index, int):
            raise PostprocessError(
                f"run {run.run_id}: request record without integer "
                f"queue_index: {record!r}"
            )
        return queue_index

    rows: list[dict[str, Any]] = []
    type_counts = {"human": 0, "tool": 0, "unknown": 0}
    first_token_source_counts: dict[str, int] = {}
    completed_count = 0
    seen: set[int] = set()
    # WP9 fallback (B4): train-interpolated proxy index over the run's
    # archived train_ledger.jsonl (loaded lazily; absent ledger -> NA note).
    proxy_index = _TrainProxyIndex(run)
    proxy_available = proxy_index.available()
    for record in sorted(run.requests, key=sort_key):
        queue_index = record["queue_index"]
        if queue_index in seen:
            raise PostprocessError(
                f"run {run.run_id}: duplicate request record for queue_index "
                f"{queue_index}"
            )
        seen.add(queue_index)
        entry = entry_by_queue_index.pop(queue_index, None)
        if entry is None:
            raise PostprocessError(
                f"run {run.run_id}: request record queue_index {queue_index} "
                "has no manifest.json counterpart (join failure, fail-closed)"
            )
        record_request_id = record.get("request_id")
        manifest_request_id = entry.get("request_id")
        if (
            record_request_id
            and manifest_request_id
            and record_request_id != manifest_request_id
        ):
            raise PostprocessError(
                f"run {run.run_id}: queue_index {queue_index} request_id "
                f"mismatch: log={record_request_id!r} manifest="
                f"{manifest_request_id!r}"
            )

        request_type = entry.get("request_type")
        if request_type not in REQUEST_TYPE_VALUES:
            request_type = "unknown"
        type_counts[request_type] += 1

        completed = bool(record.get("completed"))
        if completed:
            completed_count += 1

        notes: list[str] = []
        if not completed:
            reason = record.get("incomplete_reason") or (
                "incomplete (record carries no incomplete_reason)"
            )
            notes.append(f"incomplete_reason={reason}")
        if request_type == "unknown":
            notes.append("request_type_unknown")

        arrival = _request_int(record, "arrival_ns", queue_index)
        completion = _request_int(record, "completion_ns", queue_index)
        e2e = _request_int(record, "e2e_ns", queue_index)
        queue_ns = _request_int(record, "queue_ns", queue_index)
        prefill_ns = _request_int(record, "prefill_ns", queue_index)
        gap_ns = _request_int(record, "prefill_decode_gap_ns", queue_index)
        decode_ns = _request_int(record, "decode_ns", queue_index)

        # Ordering invariants (only on rows where the values exist).
        prefix = f"queue_index {queue_index}: "
        if arrival is not None and completion is not None:
            if arrival > completion:
                raise PostprocessError(prefix + "arrival_ns > completion_ns")
            if e2e is not None and e2e != completion - arrival:
                raise PostprocessError(
                    prefix + f"e2e_ns {e2e} != completion_ns - arrival_ns "
                    f"({completion} - {arrival})"
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
                prefix + "stage decomposition (queue+prefill+gap+decode) "
                f"!= e2e_ns ({queue_ns}+{prefill_ns}+{gap_ns}+{decode_ns} "
                f"!= {e2e})"
            )

        manifest_prefill = entry.get("prefill_length")
        manifest_decode = entry.get("decode_length")
        prefix_len = entry.get("history_tokens_before")
        if not isinstance(manifest_prefill, int) or isinstance(manifest_prefill, bool):
            manifest_prefill = None
        if not isinstance(manifest_decode, int) or isinstance(manifest_decode, bool):
            manifest_decode = None
        if not isinstance(prefix_len, int) or isinstance(prefix_len, bool):
            prefix_len = None
            notes.append("prefix_len_unavailable: manifest lacks history_tokens_before")

        def cell(value: Optional[int]) -> str:
            return REQUEST_METRICS_NA if value is None else str(value)

        # SLO B2-C (WP9) + B4 fallback: first-token tick from the C++
        # request record (FIRST_TOKEN_COMPLETE, code 8 -- null while the
        # python-side first-token markers are absent, i.e. split off).  No
        # exact value + archived train ledger -> train-interpolated proxy
        # (source=train_interpolated, display metric only -- rejected from
        # SLO judgment paths); otherwise NA.
        first_token = _request_int(record, "first_token_ns", queue_index)
        first_token_source = (
            _FIRST_TOKEN_SOURCE_EXACT if first_token is not None else None
        )
        if first_token is None and proxy_available:
            proxy_ns, proxy_note = _first_token_proxy_value(
                proxy_index, entry, record)
            if proxy_ns is not None:
                first_token = proxy_ns
                first_token_source = _FIRST_TOKEN_SOURCE_PROXY
            notes.append(proxy_note)
        elif first_token is None:
            notes.append("proxy_unavailable:no_train_ledger(results/)")
        if first_token_source is None:
            first_token_source = REQUEST_METRICS_NA
        first_token_source_counts[first_token_source] = (
            first_token_source_counts.get(first_token_source, 0) + 1)
        # Ordering invariant re-checked fail-closed on the FILLED value
        # (exact or proxy): arrival <= first_token <= completion.
        if first_token is not None and arrival is not None and completion is not None:
            if not (arrival <= first_token <= completion):
                raise PostprocessError(
                    prefix + f"first_token_ns {first_token} outside "
                    f"[arrival_ns {arrival}, completion_ns {completion}]"
                )

        turn_index = record.get("turn_index")
        if turn_index is None:
            turn_index = entry.get("turn_index")
        session_id = record.get("session_id")
        if session_id in (None, ""):
            session_id = entry.get("session_id")

        rows.append(
            {
                "queue_index": queue_index,
                "request_id": record_request_id or manifest_request_id or "",
                "session_id": "" if session_id is None else session_id,
                "turn_index": "" if turn_index is None else turn_index,
                "request_type": request_type,
                "terminal_status": "completed" if completed else "failed",
                "arrival_ns": cell(arrival),
                "prefill_start_ns": cell(
                    _request_int(record, "prefill_start_ns", queue_index)
                ),
                "prefill_end_ns": cell(
                    _request_int(record, "prefill_end_ns", queue_index)
                ),
                "decode_start_ns": cell(
                    _request_int(record, "decode_start_ns", queue_index)
                ),
                "first_token_ns": cell(first_token),
                "first_token_source": first_token_source,
                "completion_ns": cell(completion),
                "queue_ns": cell(queue_ns),
                "prefill_ns": cell(prefill_ns),
                "prefill_decode_gap_ns": cell(gap_ns),
                "decode_ns": cell(decode_ns),
                "e2e_ns": cell(e2e),
                "kv_hit_state": REQUEST_METRICS_NA,
                "restore_start_ns": REQUEST_METRICS_NA,
                "restore_complete_ns": REQUEST_METRICS_NA,
                "pre_prefill_restore_ns": REQUEST_METRICS_NA,
                "hidden_restore_ns": REQUEST_METRICS_NA,
                "exposed_restore_stall_ns": REQUEST_METRICS_NA,
                "hidden_ratio": REQUEST_METRICS_NA,
                "prefill_length": cell(manifest_prefill),
                "decode_length": cell(manifest_decode),
                "prefix_len": cell(prefix_len),
                "instructions": "; ".join(notes),
            }
        )
    if entry_by_queue_index:
        raise PostprocessError(
            f"run {run.run_id}: manifest requests without request records: "
            f"{sorted(entry_by_queue_index)[:10]}"
        )
    summary = {
        "requests": len(rows),
        "completed": completed_count,
        "failed": len(rows) - completed_count,
        "request_type_counts": type_counts,
        # WP9 B4 fallback: first_token_source 分布（exact / train_
        # interpolated / NA）——proxy 仅展示口径，不进判定。
        "first_token_source_counts": first_token_source_counts,
    }
    return rows, summary


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
        ops_by_rank[rank] = int(record.get("total_num_ops", 0))
        bytes_by_rank[rank] = int(record.get("total_local_bytes", 0))
        peak_flops = record.get("peak_flops_per_second") or peak_flops
        local_bw = record.get("local_hbm_bw_bytes_per_second") or local_bw
    rows = []
    for iteration in run.iterations:
        row = _base_row(run, labels, SOURCE_MICROBENCHMARK)
        iteration_time_ns = iteration.get("iteration_time_ns")
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
    return rows


def _planner_roofline_rows(
    run: Run, labels: dict[str, str], records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
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

    configured_groups = group_config.get("groups")
    if not isinstance(configured_groups, list) or not configured_groups:
        raise PostprocessError("group config must contain a non-empty 'groups' list")
    for entry in configured_groups:
        group_id = entry.get("group_id")
        if not group_id:
            raise PostprocessError("group config entry without group_id")
        members: list[Run] = []
        labels_by_run: dict[str, dict[str, Any]] = {}
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
                for record in run.iterations:
                    iteration_time = record.get("iteration_time_ns")
                    if iteration_time:
                        microbench.append((run, record, float(iteration_time)))
            else:
                summary = run.summary
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
        help=(
            "per-request CSV (WP1); requires exactly one full-detail service "
            "run joined against its manifest.json (fail-closed)"
        ),
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
    except PostprocessError as error:
        print(f"postprocess error: {error}", file=sys.stderr)
        return 1

    _write_csv(Path(args.out_raw), RAW_COLUMNS, raw_rows)
    _write_csv(Path(args.out_normalized), NORMALIZED_COLUMNS, normalized_rows)
    if args.out_request is not None:
        if len(runs) != 1:
            print(
                "postprocess error: request_metrics requires exactly one run "
                f"in the input logs; got {len(runs)}",
                file=sys.stderr,
            )
            return 1
        try:
            _, service_manifest, _ = _load_manifest_sidecars(runs[0])
            request_rows, request_summary = _request_metric_rows(
                runs[0], service_manifest
            )
        except PostprocessError as error:
            print(f"postprocess error: {error}", file=sys.stderr)
            return 1
        if request_rows is None:
            print(
                f"request_metrics: skipped ({request_summary['reason']}); "
                f"{args.out_request} not written"
            )
        else:
            _write_csv(Path(args.out_request), REQUEST_METRICS_COLUMNS, request_rows)
            print(
                json.dumps(
                    {"request_metrics": args.out_request, **request_summary}
                )
            )
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
