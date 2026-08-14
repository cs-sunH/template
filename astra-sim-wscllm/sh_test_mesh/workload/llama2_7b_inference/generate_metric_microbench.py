#!/usr/bin/env python3
"""Generate per-point iteration-time / utilization microbenchmark ETs.

Implementation doc sec.8: each configuration point
``(phase, tp_degree, prefill_chunk | decode_batch, kv_length, repeat_index)``
gets one small standalone Chakra ET set (sec.8.3) built with this repository's
own :func:`generate_trace.transformer_pass` operator builder (sec.8.4/8.5) —
never by expanding a full service workload (sec.8.1), and without touching
the normal service ETs in any way.

Point shapes (sec.8.4/8.5):

- prefill: ``tokens = prefill_chunk`` and
  ``kv_length = kv_length + prefill_chunk`` (the context *after* the current
  chunk, matching the service generator's span convention);
- decode: ``tokens = decode_batch`` and ``kv_length = kv_length``.

TP legality filter (sec.8.2): a tp degree is kept only when it is a positive
integer no larger than the NPU count, does not exceed ``num_heads`` (the
existing exact uneven whole-head/MLP/vocab shard rules are reused unchanged),
``hidden_size % num_heads == 0`` holds for the model, and the rank-selection
rectangle fits the mesh.

Active-rank selection rule (sec.8.3 — the other three repos replicate this
exact rule): for ``tp_degree = t`` on a ``mesh_rows x mesh_cols`` mesh, pick
the smallest ``r >= 1`` such that ``r`` divides ``t``, ``c = t // r`` satisfies
``c <= mesh_cols``, and ``r <= mesh_rows``; the active ranks are the
``r x c`` rectangle anchored at the mesh corner in row-major order,
``rank(i, j) = i * mesh_cols + j`` for ``0 <= i < r``, ``0 <= j < c``.
Every otherwise-empty (idle) rank gets one 1us root CPU timer sentinel node,
the existing ASTRA-compatibility idle sentinel mechanism (same as the WSC
generator's ``IDLE_SENTINEL_DURATION_NS``).

Boundaries (sec.8.6): per active rank, event code 5 (issue) on the first real
node and code 6 (complete) on the last real node of the single Transformer
pass; subject_id is the ``benchmark_point_id``.  The simulator resolves
``iteration_start = min(first-node issue)`` and
``iteration_end = max(last-node complete)`` across the active ranks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

SH_TEST_DIR = MODULE_DIR.parents[1]
if str(SH_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(SH_TEST_DIR))

from generate_wsc_llm_trace import (  # noqa: E402
    WscLlmTraceConfig,
    load_wsc_llm_trace_config,
)
from generate_trace import (  # noqa: E402
    ChakraAttr,
    GlobalMetadata,
    TraceBuilder,
    encode_message,
    transformer_pass,
)
from config_resolver import (  # noqa: E402
    load_hardware_config,
    materialize_runtime_configs,
)
from metrics_integration import (  # noqa: E402
    REPO_VARIANT,
    canonical_json,
    compute_trace_digest,
)
from metrics_schema import (  # noqa: E402
    EVENT_MICROBENCH_ITERATION_END,
    EVENT_MICROBENCH_ITERATION_START,
    MemoryProjection,
    MetricManifestBuilder,
    RUN_MODE_MICROBENCHMARK,
)


OUTPUT_PREFIX = "llama2_7b_microbench"
IDLE_SENTINEL_DURATION_NS = 1000
PHASES = ("prefill", "decode")


class _MicrobenchTraceBuilder(TraceBuilder):
    """TraceBuilder that drops All-Reduce collectives for tp_degree == 1.

    A single-rank All-Reduce is a semantic no-op, and ASTRA-sim's analytical
    backend never completes a 1-member collective (the GPU comm node is never
    released, which would wedge every later node in the DAG).  tp > 1 passes
    use the builder completely unchanged.
    """

    def __init__(self, *, tp_degree: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tp_degree = tp_degree

    def all_reduce(self, name: str, comm_size: int, pg_name: str) -> None:
        if self.tp_degree == 1:
            return
        super().all_reduce(name, comm_size, pg_name)

# Digest recipes for the microbenchmark sidecar manifest (deterministic):
# - trace_digest: identical to the service recipe (per-rank .et sha256 lines,
#   joined, sha256 again) so the run script's pre-launch check is shared;
# - request_mapping_digest: sha256 of the canonical JSON of the point spec
#   (the "mapping" of a microbenchmark run is exactly its point descriptor);
# - kv_event_digest: sha256 of the canonical JSON of an empty list — a
#   microbenchmark ET has no KV management events.


@dataclass(frozen=True)
class BenchPoint:
    benchmark_point_id: int
    phase: str
    tp_degree: int
    prefill_chunk: int
    decode_batch: int
    kv_length: int
    repeat_index: int
    tokens: int
    pass_kv_length: int
    ranks: tuple[int, ...]
    pg_name: str

    def spec(self) -> dict[str, Any]:
        return {
            "benchmark_point_id": self.benchmark_point_id,
            "phase": self.phase,
            "tp_degree": self.tp_degree,
            "prefill_chunk": self.prefill_chunk,
            "decode_batch": self.decode_batch,
            "kv_length": self.kv_length,
            "repeat_index": self.repeat_index,
            "tokens": self.tokens,
            "pass_kv_length": self.pass_kv_length,
            "ranks": list(self.ranks),
            "pg_name": self.pg_name,
            "rank_selection": (
                "smallest r>=1 with r|tp, c=tp/r<=mesh_cols, r<=mesh_rows; "
                "r x c row-major rectangle anchored at mesh corner (0,0)"
            ),
            "single_rank_all_reduce_dropped": self.tp_degree == 1,
        }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_microbench_ranks(
    tp_degree: int, mesh_rows: int, mesh_cols: int
) -> tuple[int, ...]:
    """Compact fixed rank rectangle for one tp degree (see module docstring)."""

    for rows in range(1, mesh_rows + 1):
        if tp_degree % rows:
            continue
        cols = tp_degree // rows
        if cols <= mesh_cols:
            return tuple(
                row * mesh_cols + col for row in range(rows) for col in range(cols)
            )
    raise ValueError(
        f"tp_degree {tp_degree} has no legal rectangle on a "
        f"{mesh_rows}x{mesh_cols} mesh"
    )


def filter_tp_degrees(
    tp_degrees: Sequence[int], config: WscLlmTraceConfig
) -> tuple[int, ...]:
    """Keep only tp degrees legal for the model and hardware (doc sec.8.2)."""

    legal: list[int] = []
    for tp in tp_degrees:
        if isinstance(tp, bool) or not isinstance(tp, int) or tp <= 0:
            continue
        if tp > config.npus_count or tp > config.num_heads:
            continue
        if config.hidden_size % config.num_heads:
            continue
        try:
            select_microbench_ranks(
                tp, config.hardware.mesh_rows, config.hardware.mesh_cols
            )
        except ValueError:
            continue
        legal.append(tp)
    return tuple(legal)


def enumerate_points(
    *,
    tp_degrees: Sequence[int],
    prefill_chunks: Sequence[int],
    decode_batches: Sequence[int],
    kv_lengths: Sequence[int],
    repeats: int,
    config: WscLlmTraceConfig,
) -> list[BenchPoint]:
    points: list[BenchPoint] = []
    next_id = 0
    for phase in PHASES:
        for tp_degree in filter_tp_degrees(tp_degrees, config):
            ranks = select_microbench_ranks(
                tp_degree, config.hardware.mesh_rows, config.hardware.mesh_cols
            )
            pg_name = str(tp_degree)
            sizes = prefill_chunks if phase == "prefill" else decode_batches
            for size in sizes:
                for kv_length in kv_lengths:
                    for repeat_index in range(repeats):
                        if phase == "prefill":
                            tokens = size
                            pass_kv_length = kv_length + size
                        else:
                            tokens = size
                            pass_kv_length = kv_length
                        points.append(
                            BenchPoint(
                                benchmark_point_id=next_id,
                                phase=phase,
                                tp_degree=tp_degree,
                                prefill_chunk=size if phase == "prefill" else 0,
                                decode_batch=size if phase == "decode" else 0,
                                kv_length=kv_length,
                                repeat_index=repeat_index,
                                tokens=tokens,
                                pass_kv_length=pass_kv_length,
                                ranks=ranks,
                                pg_name=pg_name,
                            )
                        )
                        next_id += 1
    return points


def _build_metadata(
    config: WscLlmTraceConfig, point: BenchPoint, *, rank: int
) -> GlobalMetadata:
    metadata = GlobalMetadata(version="0.0.4")
    metadata.attr.extend(
        [
            ChakraAttr(name="schema", string_val="1.0.2-chakra.0.0.4"),
            ChakraAttr(name="model", string_val=config.model_name),
            ChakraAttr(name="mlp_variant", string_val=config.mlp_variant),
            ChakraAttr(name="execution_mode", string_val="metric_microbenchmark"),
            ChakraAttr(name="npus_count", uint64_val=config.npus_count),
            ChakraAttr(name="rank", uint64_val=rank),
            ChakraAttr(
                name="benchmark_point_id", uint64_val=point.benchmark_point_id
            ),
            ChakraAttr(name="microbenchmark_phase", string_val=point.phase),
            ChakraAttr(name="tensor_parallel", uint64_val=point.tp_degree),
            ChakraAttr(name="prefill_chunk", uint64_val=point.prefill_chunk),
            ChakraAttr(name="decode_batch", uint64_val=point.decode_batch),
            ChakraAttr(name="kv_length", uint64_val=point.kv_length),
            ChakraAttr(name="repeat_index", uint64_val=point.repeat_index),
        ]
    )
    return metadata


def write_point_trace(
    config: WscLlmTraceConfig, point: BenchPoint, point_dir: Path
) -> Path:
    """Emit one point's ET set (active ranks + idle sentinels) and its
    metrics_manifest.json sidecar (run_mode=microbenchmark)."""

    point_dir.mkdir(parents=True, exist_ok=True)
    active = set(point.ranks)
    builders = {
        rank: _MicrobenchTraceBuilder(
            tp_degree=point.tp_degree,
            remote_operand_loads=config.remote_operand_loads,
        )
        if rank in active
        else TraceBuilder(remote_operand_loads=config.remote_operand_loads)
        for rank in range(config.npus_count)
    }
    events: list[tuple[int, int, int, int]] = []
    for relative_rank, rank in enumerate(point.ranks):
        first_node_id = builders[rank].next_id
        transformer_pass(
            builders[rank],
            phase=(
                f"mb_p{point.benchmark_point_id:06d}_{point.phase}"
                f"_tp{point.tp_degree}"
            ),
            tokens=point.tokens,
            kv_length=point.pass_kv_length,
            layers=config.layers,
            hidden_size=config.hidden_size,
            ffn_size=config.ffn_size,
            tensor_parallel=point.tp_degree,
            pg_name=point.pg_name,
            vocab_size=config.vocab_size,
            bytes_per_elem=config.bytes_per_elem,
            num_heads=config.num_heads,
            tensor_parallel_rank=relative_rank,
            mlp_variant=config.mlp_variant,
        )
        last_node_id = builders[rank].previous_id
        events.append(
            (
                rank,
                first_node_id,
                EVENT_MICROBENCH_ITERATION_START,
                point.benchmark_point_id,
            )
        )
        events.append(
            (
                rank,
                last_node_id,
                EVENT_MICROBENCH_ITERATION_END,
                point.benchmark_point_id,
            )
        )
    idle_ranks: list[int] = []
    for rank in range(config.npus_count):
        if rank in active:
            continue
        node_id = builders[rank].timer_gate(
            f"microbench_idle_rank_{rank:04d}_sentinel", IDLE_SENTINEL_DURATION_NS
        )
        if node_id is None:
            raise RuntimeError(f"rank {rank}: idle sentinel timer was not created")
        idle_ranks.append(rank)

    et_paths: dict[int, Path] = {}
    for rank in range(config.npus_count):
        et_path = point_dir / f"{OUTPUT_PREFIX}.{rank}.et"
        with et_path.open("wb") as output:
            encode_message(output, _build_metadata(config, point, rank=rank))
            for node in builders[rank].nodes:
                encode_message(output, node)
        et_paths[rank] = et_path

    builder = MetricManifestBuilder(
        repo_variant=REPO_VARIANT,
        run_mode=RUN_MODE_MICROBENCHMARK,
        npus_count=config.npus_count,
        mesh_rows=config.hardware.mesh_rows,
        mesh_columns=config.hardware.mesh_cols,
        node_count_by_rank={
            rank: builders[rank].node_count for rank in range(config.npus_count)
        },
        memory_projection=MemoryProjection(),
    )
    for rank, node_id, event_code, subject_id in events:
        builder.add_node_event(rank, node_id, event_code, subject_id)
    builder.set_digests(
        trace_digest=compute_trace_digest(et_paths),
        request_mapping_digest=_sha256_text(canonical_json(point.spec())),
        kv_event_digest=_sha256_text(canonical_json([])),
    )
    manifest = builder.build().to_dict()
    # Extension key (ignored by the frozen schema reader and the C++
    # collector): lets the run script and the postprocessor recover the point
    # descriptor from the manifest referenced by a run log.
    manifest["microbenchmark"] = point.spec()
    manifest["microbenchmark"]["idle_rank_sentinel"] = {
        "duration_ns": IDLE_SENTINEL_DURATION_NS,
        "idle_ranks": idle_ranks,
        "note": (
            "ASTRA compatibility sentinel for an otherwise-empty rank; the "
            "sentinel carries no boundary events and no compute work"
        ),
    }
    manifest_path = point_dir / "metrics_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    return manifest_path


def _parse_int_list(value: str, name: str) -> tuple[int, ...]:
    result = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed < 0:
            raise SystemExit(f"{name} entries must be non-negative: {value!r}")
        result.append(parsed)
    if not result:
        raise SystemExit(f"{name} must not be empty")
    return tuple(result)


def load_microbench_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        config = json.load(source)
    microbench = config.get("microbenchmark")
    if not isinstance(microbench, dict):
        raise SystemExit(f"{path} has no 'microbenchmark' block")
    return microbench


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-point microbenchmark ETs (doc sec.8)."
    )
    parser.add_argument(
        "--metrics-config",
        default=str(MODULE_DIR / "metrics_config.json"),
        help="metrics_config.json with the microbenchmark sweep block",
    )
    parser.add_argument(
        "--service-config",
        default=str(MODULE_DIR / "trace_config.csv"),
        help="service trace config CSV supplying model + hardware defaults",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="microbenchmark output root (default: sh_test_mesh/generated/metric_microbench)",
    )
    parser.add_argument("--tp-degrees", default=None, help="subset, e.g. 1,2")
    parser.add_argument("--prefill-chunks", default=None, help="subset, e.g. 128")
    parser.add_argument("--decode-batches", default=None, help="subset, e.g. 1,4")
    parser.add_argument(
        "--kv-lengths", default=None, help="subset, e.g. 128,1024"
    )
    parser.add_argument("--repeats", default=None, type=int, help="subset repeats")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="enumerate and print the points without writing any ET",
    )
    args = parser.parse_args(argv)

    microbench = load_microbench_config(Path(args.metrics_config))
    tp_degrees = (
        _parse_int_list(args.tp_degrees, "--tp-degrees")
        if args.tp_degrees is not None
        else tuple(int(v) for v in microbench["tp_degrees"])
    )
    prefill_chunks = (
        _parse_int_list(args.prefill_chunks, "--prefill-chunks")
        if args.prefill_chunks is not None
        else tuple(int(v) for v in microbench["prefill_chunks"])
    )
    decode_batches = (
        _parse_int_list(args.decode_batches, "--decode-batches")
        if args.decode_batches is not None
        else tuple(int(v) for v in microbench["decode_batches"])
    )
    kv_lengths = (
        _parse_int_list(args.kv_lengths, "--kv-lengths")
        if args.kv_lengths is not None
        else tuple(int(v) for v in microbench["kv_lengths"])
    )
    repeats = (
        args.repeats if args.repeats is not None else int(microbench.get("repeats", 1))
    )
    if repeats <= 0:
        raise SystemExit("repeats must be a positive integer")

    config = load_wsc_llm_trace_config(Path(args.service_config))
    points = enumerate_points(
        tp_degrees=tp_degrees,
        prefill_chunks=prefill_chunks,
        decode_batches=decode_batches,
        kv_lengths=kv_lengths,
        repeats=repeats,
        config=config,
    )
    if not points:
        raise SystemExit("no legal microbenchmark points after TP filtering")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else SH_TEST_DIR / "generated" / "metric_microbench"
    )
    output_dir = output_dir.resolve()

    legal_tps = tuple(dict.fromkeys(point.tp_degree for point in points))
    if args.dry_run:
        print(
            json.dumps(
                {
                    "points": len(points),
                    "legal_tp_degrees": list(legal_tps),
                    "output_dir": str(output_dir),
                    "phases": {phase: sum(1 for p in points if p.phase == phase) for phase in PHASES},
                }
            )
        )
        for point in points:
            print(json.dumps(point.spec(), separators=(",", ":")))
        return

    # One shared runtime config set; each point's ET references only its own
    # communicator group (pg id == tp degree).
    resolved_hardware = load_hardware_config(
        config.hardware_config, config.hardware_capacity_profile
    )
    runtime_paths = materialize_runtime_configs(
        hardware=resolved_hardware,
        system_template_path=config.system_template,
        inference_groups=tuple(
            (str(tp), select_microbench_ranks(tp, *(
                config.hardware.mesh_rows, config.hardware.mesh_cols
            )))
            for tp in legal_tps
        ),
        output_dir=output_dir / "runtime_config",
    )

    index: list[dict[str, Any]] = []
    for point in points:
        point_dir = output_dir / "points" / f"point_{point.benchmark_point_id:06d}"
        manifest_path = write_point_trace(config, point, point_dir)
        entry = point.spec()
        entry["point_dir"] = str(point_dir)
        entry["workload_prefix"] = str(point_dir / OUTPUT_PREFIX)
        entry["metrics_manifest"] = str(manifest_path)
        index.append(entry)

    index_path = output_dir / "microbench_index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema": 1,
                "source": "simulator_microbenchmark",
                "repo_variant": REPO_VARIANT,
                "npus_count": config.npus_count,
                "model": config.model_name,
                "hardware_label": config.hardware.label,
                "system_config": str(runtime_paths.system),
                "network_config": str(runtime_paths.network),
                "remote_memory_config": str(runtime_paths.remote_memory),
                "comm_group_config": str(runtime_paths.comm_group),
                "points": index,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "points": len(points),
                "index": str(index_path),
                "legal_tp_degrees": list(legal_tps),
            }
        )
    )


if __name__ == "__main__":
    main()
