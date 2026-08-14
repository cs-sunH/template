"""Glue between the FACE ET generator and the frozen metrics schema.

This module is strictly observational (implementation doc sec.4/6/7): it
collects request stage boundary node ids while the ET is emitted, mirrors the
planner memory deltas recorded by ``session_kv_manager`` through the read-only
:class:`metrics_schema.MemoryMetricsObserver`, and finally writes the
``metrics_manifest.json`` sidecar next to the generated ``.et`` files.

Nothing here feeds back into request mapping, scheduling, KV management, or
ET construction.  When metrics are disabled the generator never builds this
context, so the ``.et`` output stays byte-identical (doc sec.12.3).

Digest recipes (deterministic, identical with metrics on/off):

- ``trace_digest``: per rank ``<rank>:<sha256 hexdigest of the .et bytes>``
  lines joined with ``\\n``, then SHA256 of that UTF-8 text.  The run script
  recomputes exactly this recipe before launching the simulator (doc sec.10).
- ``request_mapping_digest``: SHA256 of the canonical JSON
  (``sort_keys``, compact separators) of the original manifest's ``requests``
  array, which carries the FACE prefill/decode mapping and KV placement.
- ``kv_event_digest``: SHA256 of the canonical JSON of the KV event payload
  (session-LRU: every ``KVCacheEvent`` field row in order; legacy: the final
  edge weights and remaining capacities of the local-first allocator).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from metrics_schema import (  # noqa: E402
    Arrival,
    EVENT_DECODE_END,
    EVENT_DECODE_START,
    EVENT_MEMORY_ANCHOR_COMPLETE,
    EVENT_PREFILL_END,
    EVENT_PREFILL_START,
    MemoryDelta,
    MemoryMetricsObserver,
    MemoryProjection,
    MetricManifestBuilder,
    RequestMetadata,
    RUN_MODE_SERVICE,
)
from session_kv_manager import model_weight_shard_bytes_by_tp_rank  # noqa: E402


REPO_VARIANT = "astra-sim-face"
METRICS_CONFIG_PATH = Path(__file__).resolve().parent / "metrics_config.json"

# Anchor kinds follow the doc sec.7.8 table; anything else is a nearest-stage
# fallback and is tagged anchor_quality=stage_boundary in the manifest.
EXACT_ANCHOR_KINDS = frozenset(
    {"tick_zero", "prefill_start", "transfer_complete", "completion"}
)

__all__ = [
    "EVENT_DECODE_END",
    "EVENT_DECODE_START",
    "EVENT_MEMORY_ANCHOR_COMPLETE",
    "EVENT_PREFILL_END",
    "EVENT_PREFILL_START",
    "MemoryActionRecorder",
    "PlannerLutStatsAccumulator",
    "ServiceMetrics",
    "canonical_json",
    "compute_trace_digest",
    "resolve_metrics_detail",
    "write_planner_lut_stats",
]


def kv_bin_power_of_two(value: int) -> int:
    """Bin a KV length into its power-of-two ceiling (0 stays bin 0)."""

    value = int(value)
    if value <= 0:
        return 0
    return 1 << (value - 1).bit_length()


class PlannerLutStatsAccumulator:
    """Streaming planner-iteration aggregates (doc sec.8.8).

    The service planner notifies one LUT lookup per planning iteration through
    :meth:`record_lut_iteration`; only per-cell count/sum/min/max are kept, so
    memory stays O(cells) regardless of iteration count.  Cells are keyed by
    ``(phase, tp_degree, batch, kv_bin)`` where phase is ``prefill`` /
    ``decode`` / ``mixed`` (a mixed iteration carries both a prefill chunk and
    a decode batch in the FACE LUT model), batch is the prefill chunk for
    prefill cells and the decode batch otherwise, and kv_bin is the
    power-of-two ceiling of the decode KV length.  These records are a
    planner-LUT proxy (``source=planner_lut``), never the paper's primary
    iteration-time source.
    """

    def __init__(self) -> None:
        # (phase, tp_degree, batch, kv_bin, prefill_chunk) -> [count,sum,min,max]
        self._cells: dict[tuple[str, int, int, int, int], list[int]] = {}

    def record_lut_iteration(
        self, lut_entry: Any, start_ns: int, end_ns: int
    ) -> None:
        p_chunk = int(lut_entry.p_chunk)
        d_batch = int(lut_entry.d_batch)
        if p_chunk > 0 and d_batch > 0:
            phase = "mixed"
        elif p_chunk > 0:
            phase = "prefill"
        else:
            phase = "decode"
        batch = p_chunk if phase == "prefill" else d_batch
        key = (
            phase,
            int(lut_entry.instance_size),
            batch,
            kv_bin_power_of_two(int(lut_entry.d_token)),
            p_chunk,
        )
        iteration_time_ns = int(end_ns) - int(start_ns)
        cell = self._cells.get(key)
        if cell is None:
            self._cells[key] = [1, iteration_time_ns, iteration_time_ns, iteration_time_ns]
        else:
            cell[0] += 1
            cell[1] += iteration_time_ns
            cell[2] = min(cell[2], iteration_time_ns)
            cell[3] = max(cell[3], iteration_time_ns)

    def to_records(self, *, repo_variant: str = REPO_VARIANT) -> list[dict[str, Any]]:
        records = []
        for (phase, tp_degree, batch, kv_bin, p_chunk), cell in sorted(
            self._cells.items()
        ):
            records.append(
                {
                    "schema": 1,
                    "type": "planner_lut_iteration_stats",
                    "source": "planner_lut",
                    "repo_variant": repo_variant,
                    "phase": phase,
                    "tp_degree": tp_degree,
                    "batch": batch,
                    "kv_bin": kv_bin,
                    "prefill_chunk_tokens": p_chunk,
                    "count": cell[0],
                    "sum_iteration_time_ns": cell[1],
                    "min_iteration_time_ns": cell[2],
                    "max_iteration_time_ns": cell[3],
                }
            )
        return records


def write_planner_lut_stats(
    accumulator: PlannerLutStatsAccumulator,
    *,
    output_dir: Path,
    repo_variant: str = REPO_VARIANT,
) -> Path:
    """Write the planner_lut_stats.json sidecar and echo every record as a
    single-line ``[METRIC]`` JSON record (doc sec.8.8/11.1)."""

    records = accumulator.to_records(repo_variant=repo_variant)
    sidecar = output_dir / "planner_lut_stats.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema": 1,
                "source": "planner_lut",
                "repo_variant": repo_variant,
                "records": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for record in records:
        print("[METRIC] " + json.dumps(record, separators=(",", ":")))
    return sidecar


def canonical_json(payload: Any) -> str:
    """Deterministic JSON serialization used by every metrics digest."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_trace_digest(et_paths_by_rank: Mapping[int, Path]) -> str:
    """SHA256 over the sorted per-rank ``.et`` SHA256 lines (see module doc)."""

    per_rank = []
    for rank, path in sorted(et_paths_by_rank.items()):
        per_rank.append(f"{rank}:{hashlib.sha256(path.read_bytes()).hexdigest()}")
    return _sha256_hex("\n".join(per_rank))


def request_mapping_digest(request_records: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_hex(canonical_json(list(request_records)))


def kv_event_digest(kv_payload: Any) -> str:
    return _sha256_hex(canonical_json(kv_payload))


def kv_event_payload_session_lru(events: Sequence[Any]) -> list[list[Any]]:
    """One row per KVCacheEvent, in event order, mirroring kv_cache_events.csv."""

    return [
        [
            event.event_index,
            event.planner_time_ns,
            event.phase,
            event.event_type,
            event.reason,
            event.trigger_request_id,
            event.session_id,
            event.source_instance_index,
            event.target_instance_index,
            event.context_tokens,
            event.total_bytes,
            list(event.shard_bytes),
            event.last_completion_ns,
            list(event.instance_remaining_before_bytes),
            list(event.instance_remaining_after_bytes),
            list(event.insufficient_ranks),
        ]
        for event in events
    ]


def kv_event_payload_legacy(plan: Any) -> dict[str, Any]:
    """Legacy local-first allocator outcome (there is no KV event log)."""

    return {
        "policy": "decode_local_first_then_weighted_distance_and_capacity",
        "final_edge_weights": [list(edge) for edge in plan.final_edge_weights],
        "final_remaining_capacity_bytes": list(plan.final_remaining_capacity_bytes),
    }


def load_chiplets_per_npu() -> int:
    try:
        with METRICS_CONFIG_PATH.open(encoding="utf-8") as source:
            config = json.load(source)
        return int(config.get("memory", {}).get("chiplets_per_npu", 4))
    except (OSError, ValueError, TypeError):
        return 4


def resolve_metrics_detail(
    cli_value: Optional[str], environ: Mapping[str, str]
) -> str:
    """Resolve the generator metrics switch: CLI > METRICS_DETAIL >
    ENABLE_METRICS > default on/full (aligned with run_sh_test_aware.sh)."""

    detail = cli_value if cli_value is not None else environ.get("METRICS_DETAIL")
    if detail is None:
        enable = environ.get("ENABLE_METRICS", "1").strip().lower()
        detail = "off" if enable in {"0", "off", "false", "no"} else "full"
    detail = detail.strip().lower()
    if detail not in {"off", "summary", "full"}:
        raise SystemExit(
            f"invalid metrics detail {detail!r}; expected off|summary|full"
        )
    return detail


class MemoryActionRecorder:
    """Keeps the planner memory delta sequence while feeding the observer.

    The KV manager calls :meth:`record` only *after* it applied the identical
    state change (doc sec.7.2); the recorder assigns the sequence index,
    forwards the delta to the read-only observer, and stores it for the
    manifest ``memory_actions`` replay stream (doc sec.7.8).
    """

    def __init__(self, observer: MemoryMetricsObserver) -> None:
        self.observer = observer
        self.deltas: list[MemoryDelta] = []

    def initialize_rank(self, rank: int, capacity_bytes: int) -> None:
        self.observer.initialize_rank(rank, capacity_bytes)

    def record(
        self,
        *,
        planner_time_ns: int,
        anchor_kind: str,
        request_id: Optional[str],
        session_id: Optional[str],
        rank: int,
        allocation_key: str,
        weight_delta_bytes: int = 0,
        resident_kv_delta_bytes: int = 0,
        reserved_kv_delta_bytes: int = 0,
        cause: str = "",
    ) -> None:
        delta = MemoryDelta(
            sequence_index=len(self.deltas),
            planner_time_ns=planner_time_ns,
            anchor_kind=anchor_kind,
            trigger_queue_index=None,
            request_id=request_id,
            session_id=session_id,
            rank=rank,
            allocation_key=allocation_key,
            weight_delta_bytes=weight_delta_bytes,
            resident_kv_delta_bytes=resident_kv_delta_bytes,
            reserved_kv_delta_bytes=reserved_kv_delta_bytes,
            cause=cause,
        )
        self.observer.record_delta(delta)
        self.deltas.append(delta)


def _build_request_metadata(config: Any, plan: Any) -> list[RequestMetadata]:
    """Manifest request records (doc sec.4.2/4.4) for the whole queue.

    Turn-0 arrivals are absolute session arrivals; later turns reference the
    same session's previous turn request by ``queue_index`` so the simulator
    resolves them from the parent's *actual* completion (doc sec.3.1).
    """

    queue_by_session_turn = {
        (spec.session_id, spec.turn_index): index
        for index, spec in enumerate(config.request_queue)
    }
    group_by_index = dict(enumerate(config.inference_groups))
    plans_by_queue = {request_plan.queue_index: request_plan for request_plan in plan.requests}
    metadata: list[RequestMetadata] = []
    for index, spec in enumerate(config.request_queue):
        request_plan = plans_by_queue.get(index)
        if request_plan is None:
            raise RuntimeError(
                f"request queue_index {index} has no plan; the metrics manifest "
                "requires every queued request to be planned"
            )
        if spec.turn_index == 0:
            if spec.session_arrival_time_ns is None:
                raise RuntimeError("first request lost its session arrival time")
            arrival = Arrival.absolute(spec.session_arrival_time_ns)
        else:
            parent_index = queue_by_session_turn.get(
                (spec.session_id, spec.turn_index - 1)
            )
            if parent_index is None or spec.inter_request_interval_ns is None:
                raise RuntimeError(
                    f"request {spec.request_id} lost its previous-turn parent"
                )
            arrival = Arrival.after_request(
                parent_index, spec.inter_request_interval_ns
            )
        metadata.append(
            RequestMetadata(
                queue_index=index,
                request_id=request_plan.request_id,
                session_id=request_plan.session_id,
                turn_index=request_plan.turn_index,
                arrival=arrival,
                prefill_instance=request_plan.prefill_instance_index,
                prefill_ranks=tuple(
                    group_by_index[request_plan.prefill_instance_index].ranks
                ),
                decode_instance=request_plan.decode_instance_index,
                decode_ranks=tuple(
                    group_by_index[request_plan.decode_instance_index].ranks
                ),
            )
        )
    return metadata


class ServiceMetrics:
    """Per-generation metrics context for one FACE service trace run."""

    def __init__(self, detail: str, *, chiplets_per_npu: Optional[int] = None) -> None:
        if detail not in {"summary", "full"}:
            raise ValueError("ServiceMetrics requires detail summary|full")
        self.detail = detail
        self.chiplets_per_npu = (
            load_chiplets_per_npu() if chiplets_per_npu is None else chiplets_per_npu
        )
        self.memory = MemoryActionRecorder(
            MemoryMetricsObserver(self.chiplets_per_npu)
        )
        # (rank, node_id, event_code, subject_id) in emission order.
        self.events: list[tuple[int, int, int, int]] = []
        self.weight_preload_recorded = False

    def add_event(
        self, rank: int, node_id: int, event_code: int, subject_id: int
    ) -> None:
        self.events.append((rank, node_id, event_code, subject_id))

    def record_weight_preload(self, config: Any, plan: Any) -> None:
        """Legacy path only: the KV manager never runs there, so the generator
        itself mirrors the preloaded per-rank weight shards (anchor tick 0)."""

        if self.weight_preload_recorded:
            raise RuntimeError("weight preload recorded twice")
        self.weight_preload_recorded = True
        for instance in plan.topology.instances:
            shards = model_weight_shard_bytes_by_tp_rank(
                config.model, len(instance.ranks)
            )
            for relative_rank, rank in enumerate(instance.ranks):
                self.memory.initialize_rank(
                    rank, config.hardware.local_hbm_capacity_bytes
                )
        for instance in plan.topology.instances:
            shards = model_weight_shard_bytes_by_tp_rank(
                config.model, len(instance.ranks)
            )
            for relative_rank, rank in enumerate(instance.ranks):
                self.memory.record(
                    planner_time_ns=0,
                    anchor_kind="tick_zero",
                    request_id=None,
                    session_id=None,
                    rank=rank,
                    allocation_key=f"weight:{rank}",
                    weight_delta_bytes=int(shards[relative_rank]),
                    cause="model_weight_preload",
                )

    def write_manifest(
        self,
        *,
        output_dir: Path,
        config: Any,
        plan: Any,
        node_count_by_rank: Mapping[int, int],
        et_paths_by_rank: Mapping[int, Path],
        request_records: Sequence[Mapping[str, Any]],
        kv_digest_payload: Any,
    ) -> Path:
        """Assemble, validate (doc sec.4.5), and write metrics_manifest.json."""

        if not self.memory.deltas:
            raise RuntimeError(
                "no memory deltas were recorded; the session KV manager must "
                "run under set_metrics_observer() or record_weight_preload() "
                "must be called before writing the metrics manifest"
            )
        request_id_to_queue = {
            spec.request_id: index for index, spec in enumerate(config.request_queue)
        }
        builder = MetricManifestBuilder(
            repo_variant=REPO_VARIANT,
            run_mode=RUN_MODE_SERVICE,
            npus_count=config.npus_count,
            mesh_rows=config.hardware.mesh_rows,
            mesh_columns=config.hardware.mesh_cols,
            node_count_by_rank=dict(node_count_by_rank),
            memory_projection=MemoryProjection(
                chiplets_per_npu=self.chiplets_per_npu
            ),
        )
        for metadata in _build_request_metadata(config, plan):
            builder.add_request(metadata)
        for rank, node_id, event_code, subject_id in self.events:
            builder.add_node_event(rank, node_id, event_code, subject_id)
        builder.set_digests(
            trace_digest=compute_trace_digest(et_paths_by_rank),
            request_mapping_digest=request_mapping_digest(request_records),
            kv_event_digest=kv_event_digest(kv_digest_payload),
        )
        for delta in self.memory.deltas:
            action = delta.to_dict()
            if delta.request_id is not None:
                action["trigger_queue_index"] = request_id_to_queue.get(
                    delta.request_id
                )
            action["anchor_quality"] = (
                "exact"
                if delta.anchor_kind in EXACT_ANCHOR_KINDS
                else "stage_boundary"
            )
            builder.add_memory_action(action)
        memory_result = self.memory.observer.finalize()
        for rank, rank_result in sorted(memory_result.ranks.items()):
            builder.add_planner_memory_peak(rank_result.to_dict())
        manifest = builder.build()
        manifest_path = output_dir / "metrics_manifest.json"
        # Compact serialization: the memory action stream is machine-read
        # (C++ MetricCollector / post-processing), and pretty-printing it
        # would grow the sidecar by several times on large workloads.
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return manifest_path
