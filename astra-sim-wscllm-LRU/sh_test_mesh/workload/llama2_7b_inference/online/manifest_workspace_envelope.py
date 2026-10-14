"""Derive a fixed LRU manifest fit margin using only C++ capacity formulas.

P tasks come from the production prefill emitter. Every legal D train is a
subset of the complete per-request decode progressions, including capped,
first-token and remainder trains. The C++ bound includes allowed templates
before SRAM feasibility filtering; a largest-task candidate maximum would not
prove this coverage. No timing or workspace arithmetic is implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping

from online.die_model_adapter import DieModelContext, OP_KINDS, build_die_spec
from online.die_workspace_envelope import DieWorkspaceEnvelopeClient, WorkspaceEnvelopeError


@dataclass(frozen=True)
class ManifestWorkspaceCertificate:
    fingerprint: str
    hardware_fingerprint: str
    bytes_by_rank: Mapping[int, int]
    canonical_json: str

    def without_proof(self) -> "ManifestWorkspaceCertificate":
        """Keep the fixed admission policy after its larger proof is saved."""
        return ManifestWorkspaceCertificate(self.fingerprint, self.hardware_fingerprint,
                                            self.bytes_by_rank, "")


def _u64(value: Any, field: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < int(positive) or value >= 1 << 64:
        raise WorkspaceEnvelopeError(f"{field} must be an exact uint64 integer")
    return value


def derive_manifest_workspace_envelope(manifest: dict, config: Any, *,
                                       library_path=None) -> ManifestWorkspaceCertificate:
    """Compile a capacity-only certificate before any KV or graph mutation.

    This coverage is specific to the LRU P/D producer. Arbitrary task graphs,
    recompute producers and direct concurrent COMP bypasses require separate
    coverage. The actual C++ Plan is checked against the installed rank bound.
    """
    from generate_wsc_llm_trace import _emit_prefill_stage
    from online.graph_batch_builder import GraphBatchBuilder, _prefix_of

    context = DieModelContext.from_service(manifest, config)
    if context is None or context.profile != "coarse_die_v2":
        raise WorkspaceEnvelopeError("manifest workspace certificate requires coarse_die_v2")
    if config.trace_granularity != "request_aggregated":
        raise WorkspaceEnvelopeError("unsupported workspace manifest producer")
    requests = manifest.get("requests")
    if not isinstance(requests, list):
        raise WorkspaceEnvelopeError("manifest.requests must be a finite list")
    identities = set()
    for i, row in enumerate(requests):
        if not isinstance(row, dict) or not isinstance(row.get("request_id"), str):
            raise WorkspaceEnvelopeError(f"manifest.requests[{i}] lacks request identity")
        if row["request_id"] in identities:
            raise WorkspaceEnvelopeError("manifest request identities must be unique")
        identities.add(row["request_id"])
        for key in ("prefill_length", "decode_length", "history_tokens_before",
                    "prefill_context_tokens"):
            _u64(row.get(key), f"manifest.requests[{i}].{key}",
                 positive=key in {"prefill_length", "decode_length"})
        if row["prefill_context_tokens"] != row["history_tokens_before"] + row["prefill_length"]:
            raise WorkspaceEnvelopeError("manifest prefill context contradicts history and length")
        if row["prefill_context_tokens"] + row["decode_length"] >= 1 << 64:
            raise WorkspaceEnvelopeError("manifest decode progression overflows uint64")
    _u64(config.layers, "config.layers", positive=True)
    groups = tuple(config.inference_groups)
    floors = {rank: 0 for group in groups for rank in group.ranks}
    if set(floors) != set(range(config.npus_count)):
        raise WorkspaceEnvelopeError("manifest workspace groups must cover every configured rank")
    records = []

    with DieWorkspaceEnvelopeClient(context.die_config_path, library_path) as client:
        hardware_fingerprint = client.hardware_fingerprint

        def include(rank: int, spec: dict, source: str) -> None:
            bound = client.query_capacity_bound(spec)
            floors[rank] = max(floors[rank], bound.hbm_workspace_upper_bound_bytes)
            records.append({
                "rank": rank, "source": source,
                "capacity_model_revision": bound.capacity_model_revision,
                "normalized_task_fingerprint": bound.normalized_task_fingerprint,
                "normalized_task": bound.normalized_task,
                "hbm_workspace_upper_bound_bytes": bound.hbm_workspace_upper_bound_bytes,
            })

        decode_spans = [{
            "request_id_prefix": row["request_id"], "first_index": 0,
            "run": {"q_tokens": 1, "kv_first": row["prefill_context_tokens"] + 1,
                    "kv_stride": 1, "count": row["decode_length"]},
        } for row in requests]
        decode_passes = max((row["decode_length"] for row in requests), default=1)
        for group in groups:
            if group.phase_role == "prefill":
                for row in requests:
                    # A fresh graph per request prevents stale earlier spans
                    # from contaminating the next request's capacity proof.
                    graph = GraphBatchBuilder(config)
                    graph.set_die_context(context)
                    graph.begin_batch()
                    mark = graph._mark()
                    _emit_prefill_stage(config=config, builders=graph.builders,
                                        group=group, prefix=_prefix_of(row),
                                        stage="current_prefill",
                                        tokens=row["prefill_length"],
                                        initial_context_tokens=row["history_tokens_before"],
                                        die_ctx=context)
                    graph._collect(mark)
                    seen = set()
                    for node in graph.batch["nodes"]:
                        spec = node.get("compute", {}).get("die_op_spec")
                        if spec is not None:
                            include(node["rank"], spec, "production_current_prefill")
                            seen.add(node["rank"])
                    if seen != set(group.ranks):
                        raise WorkspaceEnvelopeError("prefill emitter omitted a configured rank")
            elif group.phase_role == "decode":
                if not requests:
                    continue
                for relative_rank, rank in enumerate(group.ranks):
                    context.begin_emission(phase="decode", tensor_parallel=len(group.ranks),
                                           tensor_parallel_rank=relative_rank,
                                           span_namespace="manifest_workspace_decode",
                                           pass_spans=[(1, 1)])
                    for op in sorted(OP_KINDS - {"local_region"}):
                        spec = build_die_spec(op, context, decode_spans,
                                              layer_count=config.layers,
                                              weight_passes=decode_passes,
                                              fusion_policy="auto")
                        include(rank, spec, "complete_manifest_decode_progressions")
            else:
                raise WorkspaceEnvelopeError("workspace manifest has unsupported group phase")

    payload = {
        "schema": "lru_manifest_workspace_certificate_v1",
        "coverage": "serialized_current_prefill_and_decode_subset_v1",
        "hardware_fingerprint": hardware_fingerprint,
        "capacity_model_revision": "workspace_lifetimes_v1",
        "manifest_requests_sha256": hashlib.sha256(json.dumps(
            requests, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest(),
        "ranks": [{"rank": rank, "prospective_workspace_bytes": floors[rank]}
                  for rank in sorted(floors)],
        "proof_tasks": records,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False)
    return ManifestWorkspaceCertificate(hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                                        hardware_fingerprint, MappingProxyType(floors), canonical)
