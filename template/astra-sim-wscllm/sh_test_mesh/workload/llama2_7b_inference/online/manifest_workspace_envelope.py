"""Finite WSCLLM train-shape coverage; all workspace bytes come from C++.

One train contains at most one prefill queue head, plus decode spans from
currently active requests. The prefill head queries at most chunk*cap tokens
(or its complete context without a cap). Each decode request contributes at
most min(decode_length, cap) q=1 queries. Replacing their KV lengths with each
request's final context only increases the C++ lifetime counts. Including all
manifest decode requests also covers arbitrary membership and dynamic placement.
First-step/remainder, restored history and base recompute are subsets. This
certificate relies on workspace_lifetimes_v2's sequential layer/pass reuse and
the original one-GPU-COMP-per-rank gate; it is not a general graph certificate.

2026-10-20(方案二直接替换):本文件按 astra-sim-face-LRU 同名实现整体回迁
(蓝本口径:合成 train span 超集 + train 上限),推导位置在 online_service
(FL 结构);本仓此前无任何推导/地板实现。
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from online.coarse_workspace_envelope import (
    DieWorkspaceEnvelopeClient, Pathish, WorkspaceEnvelopeError,
)
from online.die_model_adapter import DieModelContext, OP_KINDS, build_die_spec

CAPACITY_REVISION = "workspace_lifetimes_v2"
COVERAGE_REVISION = "wscllm_train_one_prefill_all_request_decode_superset_v1"


def _u64(value: Any, field: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < int(positive) or value >= 1 << 64:
        raise WorkspaceEnvelopeError(f"{field} must be an exact uint64 integer")
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _config_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {field.name: _config_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        # wscllm 适配(2026-10-20):本仓 config 携带 int 键映射
        # (queue_position_by_index),FL 蓝本的纯 str 键校验会在此 fail。
        # int 键以 "int:" 命名空间字符串化,与十进制字面 str 键永不混淆;
        # 转换后仍冲突(理论上的 "int:1" 字面键)即 fail-closed。
        converted = {}
        for key, child in value.items():
            if isinstance(key, str):
                identity = key
            elif type(key) is int:
                identity = f"int:{key}"
            else:
                raise WorkspaceEnvelopeError("configuration object keys must be strings")
            if identity in converted:
                raise WorkspaceEnvelopeError(
                    "configuration object keys collide after identity conversion")
            converted[identity] = _config_value(child)
        return converted
    if isinstance(value, (tuple, list)):
        return [_config_value(child) for child in value]
    if value is None or type(value) in (int, float, bool, str):
        return value
    raise WorkspaceEnvelopeError(f"unsupported configuration identity type {type(value).__name__}")


@dataclass(frozen=True)
class ManifestWorkspaceCertificate:
    fingerprint: str
    hardware_fingerprint: str
    bytes_by_rank: Mapping[int, int]
    canonical_json: str
    capacity_model_revision: str
    coverage_revision: str
    train_max_iter: int
    manifest_sha256: str
    config_sha256: str
    resolved_hardware_sha256: str

    def __post_init__(self):
        _u64(self.train_max_iter, "train_max_iter")
        copied = {}
        for rank, value in self.bytes_by_rank.items():
            copied[_u64(rank, "certificate rank")] = _u64(value, "certificate workspace")
        object.__setattr__(self, "bytes_by_rank", MappingProxyType(copied))
        if (self.capacity_model_revision != CAPACITY_REVISION or
                self.coverage_revision != COVERAGE_REVISION):
            raise WorkspaceEnvelopeError("unsupported workspace certificate coverage")
        if self.canonical_json and _sha(self.canonical_json) != self.fingerprint:
            raise WorkspaceEnvelopeError("workspace certificate proof fingerprint mismatch")

    def without_proof(self) -> "ManifestWorkspaceCertificate":
        return replace(self, canonical_json="")


def validate_manifest_workspace_certificate(certificate, manifest, config, train_max_iter, context):
    """Reject a stale producer identity before installing any KV state."""
    if not isinstance(certificate, ManifestWorkspaceCertificate):
        raise ValueError("invalid prospective workspace certificate type")
    if certificate.train_max_iter != train_max_iter:
        raise ValueError("workspace certificate train cap differs from producer")
    if certificate.manifest_sha256 != _sha(_canonical(dict(manifest))):
        raise ValueError("workspace certificate manifest differs from producer")
    if certificate.config_sha256 != _sha(_canonical(_config_value(config))):
        raise ValueError("workspace certificate configuration differs from producer")
    if context is None or certificate.resolved_hardware_sha256 != context.config_hash:
        raise ValueError("workspace certificate resolved hardware differs from producer")


def _checked_requests(manifest):
    requests = manifest.get("requests")
    if not isinstance(requests, list):
        raise WorkspaceEnvelopeError("manifest.requests must be a finite list")
    ids, queue_indices, sessions = set(), set(), {}
    for index, row in enumerate(requests):
        if not isinstance(row, dict):
            raise WorkspaceEnvelopeError(f"manifest.requests[{index}] must be an object")
        rid, sid = row.get("request_id"), row.get("session_id")
        if not isinstance(rid, str) or not rid or rid in ids:
            raise WorkspaceEnvelopeError("manifest request identity is missing or duplicated")
        if not isinstance(sid, str) or not sid:
            raise WorkspaceEnvelopeError("manifest session identity is missing")
        ids.add(rid)
        for key in ("queue_index", "turn_index", "prefill_length", "decode_length",
                    "history_tokens_before", "prefill_context_tokens", "final_context_tokens"):
            _u64(row.get(key), f"manifest.requests[{index}].{key}",
                 positive=key in {"prefill_length", "decode_length"})
        if row["queue_index"] in queue_indices:
            raise WorkspaceEnvelopeError("manifest queue indices must be unique")
        queue_indices.add(row["queue_index"])
        if row["prefill_context_tokens"] != row["history_tokens_before"] + row["prefill_length"]:
            raise WorkspaceEnvelopeError("manifest prefill context contradicts history and length")
        if row["final_context_tokens"] != row["prefill_context_tokens"] + row["decode_length"]:
            raise WorkspaceEnvelopeError("manifest final context contradicts decode length")
        sessions.setdefault(sid, []).append(row)
    if queue_indices != set(range(len(requests))):
        raise WorkspaceEnvelopeError("manifest queue indices must form a contiguous range")
    for sid, rows in sessions.items():
        ordered = sorted(rows, key=lambda row: row["queue_index"])
        if [row["turn_index"] for row in ordered] != list(range(len(ordered))):
            raise WorkspaceEnvelopeError(f"{sid}: turn indices must be contiguous from zero")
        if ordered[0]["history_tokens_before"] != 0:
            raise WorkspaceEnvelopeError(f"{sid}: initial history has no finite-manifest source")
        for previous, current in zip(ordered, ordered[1:]):
            if current["history_tokens_before"] != previous["final_context_tokens"]:
                raise WorkspaceEnvelopeError(f"{sid}: manifest turn history chain mismatch")
    return requests


def _capacity_task(spec):
    # Schema conversion only. Legacy compact runs require positive stride;
    # the coarse schema also represents the constant-KV shape superset.
    dims, tp, spans = spec["model_dims"], spec["tp_slice"], spec["spans"]
    rows = sum(span["run"]["q_tokens"] * span["run"]["count"]
              if "run" in span else span["q_tokens"] for span in spans)
    _u64(rows, "capacity task query rows", positive=True)
    return {"schema": "coarse_die_task_v2", "op_kind": spec["op_kind"],
        "phase": spec["phase"], "tensor_shapes": {
            "m": rows, "hidden": dims["hidden"], "ffn": tp["ffn_local"],
            "vocab": tp["vocab_local"], "q_heads": tp["q_heads_local"],
            "kv_heads": tp["kv_heads_local"], "head_dim": dims["head_dim"],
            "mlp_variant": dims["mlp_variant"], "norm_type": dims["norm_type"],
            "position_encoding": dims["position_encoding"]},
        "spans": spans, "tp_slice": tp, "storage_dtype": spec["dtype"],
        "layer_instances": spec["layer_count"], "weight_passes": spec["weight_passes"],
        "fusion_policy": spec["fusion_policy"]}


def derive_manifest_workspace_envelope(
    manifest: Mapping[str, Any], config: Any, *, train_max_iter: int,
    context: DieModelContext | None = None, library_path: Pathish | None = None,
) -> ManifestWorkspaceCertificate:
    """Certify the frozen loaded producer before graph/KV admission mutates it."""
    _u64(train_max_iter, "train_max_iter")
    if not isinstance(manifest, Mapping):
        raise WorkspaceEnvelopeError("manifest must be a mapping")
    requests = _checked_requests(manifest)
    from plan_materializer import _derive_manifest_requests
    expected_rows = _derive_manifest_requests(config, {})
    if (len(requests) != len(expected_rows) or any(
            any(row.get(key) != expected[key] for key in expected)
            for row, expected in zip(requests, expected_rows))):
        raise WorkspaceEnvelopeError("manifest must match the complete loaded request queue")
    if config.trace_granularity != "request_aggregated":
        raise WorkspaceEnvelopeError("workspace certificate requires request_aggregated producer")
    if config.kv_cache_policy not in {"session_lru_recompute", "session_lru_tiered"}:
        raise WorkspaceEnvelopeError("unsupported wscllm workspace producer policy")
    for key in ("npus_count", "layers", "hidden_size", "ffn_size", "num_heads",
                "vocab_size", "bytes_per_elem", "prefill_chunk_size"):
        _u64(getattr(config, key), "config." + key, positive=True)
    actual_context = DieModelContext.from_service(dict(manifest), config)
    if actual_context is None or actual_context.profile != "coarse_die_v2":
        raise WorkspaceEnvelopeError("manifest workspace certificate requires coarse_die_v2")
    if context is not None and (
            context.profile != actual_context.profile or
            context.config_hash != actual_context.config_hash or
            context.model_dims != actual_context.model_dims or
            context.dtype != actual_context.dtype):
        raise WorkspaceEnvelopeError("workspace context differs from loaded producer")
    # Use a fresh context so certification never mutates the caller's emission state.
    context = actual_context
    groups = tuple(config.inference_groups)
    flattened = []
    for group in groups:
        if not group.ranks:
            raise WorkspaceEnvelopeError("workspace group cannot be empty")
        flattened.extend(_u64(rank, "group rank") for rank in group.ranks)
    if sorted(flattened) != list(range(config.npus_count)):
        raise WorkspaceEnvelopeError("workspace groups must partition all configured ranks")
    tp_degrees = {len(group.ranks) for group in groups}
    if len(tp_degrees) != 1:
        raise WorkspaceEnvelopeError("wscllm certificate requires equal TP groups")
    tp = next(iter(tp_degrees))
    config_identity = _config_value(config)
    config_sha = _sha(_canonical(config_identity))
    manifest_sha = _sha(_canonical(dict(manifest)))
    spans = []
    if requests:
        max_context = max(row["prefill_context_tokens"] for row in requests)
        p_queries = max_context if train_max_iter == 0 else min(
            max_context, config.prefill_chunk_size * train_max_iter)
        spans.append({"request_id": "workspace-envelope:prefill-head",
                      "q_tokens": p_queries, "kv_tokens": max_context,
                      "q_start_pos": max_context - p_queries, "kv_start_pos": 0})
        for row in sorted(requests, key=lambda row: row["queue_index"]):
            count = row["decode_length"] if train_max_iter == 0 else min(
                row["decode_length"], train_max_iter)
            spans.append({"request_id_prefix": "workspace-envelope:decode:" + row["request_id"],
                          "first_index": 0, "run": {"q_tokens": 1,
                          "kv_first": row["final_context_tokens"], "kv_stride": 0,
                          "count": count}})
    floors = {rank: 0 for rank in range(config.npus_count)}
    records = []
    with DieWorkspaceEnvelopeClient(context.die_config_path, library_path) as client:
        hardware_fingerprint = client.hardware_fingerprint
        for relative_rank in range(tp):
            physical_ranks = [group.ranks[relative_rank] for group in groups]
            if not spans:
                continue
            for phase in ("prefill", "decode"):
                context.begin_emission(phase=phase, tensor_parallel=tp,
                    tensor_parallel_rank=relative_rank,
                    span_namespace="workspace-envelope", pass_spans=[(1, 1)])
                for op in sorted(OP_KINDS - {"local_region"}):
                    spec = build_die_spec(op, context, spans,
                        layer_count=1 if op == "logits" else config.layers,
                        weight_passes=1,
                        fusion_policy="auto" if op in {"attention_region", "ffn_region"} else "none")
                    bound = client.query_capacity_bound(_capacity_task(spec))
                    for rank in physical_ranks:
                        floors[rank] = max(floors[rank], bound.hbm_workspace_upper_bound_bytes)
                    records.append({"relative_tp_rank": relative_rank,
                        "physical_ranks": physical_ranks,
                        "request_json": bound.request_json,
                        "request_sha256": _sha(bound.request_json),
                        "response_json": bound.response_json,
                        "response_sha256": _sha(bound.response_json)})
        library_sha = hashlib.sha256(client.library_path.read_bytes()).hexdigest()
    online = Path(__file__).resolve().parent
    source_files = [online / "manifest_workspace_envelope.py",
                    online / "coarse_workspace_envelope.py",
                    online / "die_model_adapter.py", online / "graph_batch_builder.py",
                    online / "wsc_llm_online_scheduler.py", online.parent / "generate_wsc_llm_trace.py", online.parent / "plan_materializer.py"]
    payload = {"schema": "wscllm_manifest_workspace_certificate_v1",
        "coverage_revision": COVERAGE_REVISION,
        "capacity_model_revision": CAPACITY_REVISION,
        "train_max_iter": train_max_iter, "prefill_chunk_size": config.prefill_chunk_size,
        "manifest_sha256": manifest_sha, "config_sha256": config_sha,
        "config_identity": config_identity, "hardware_fingerprint": hardware_fingerprint,
        "resolved_hardware_sha256": context.config_hash, "library_sha256": library_sha,
        "source_sha256": {str(path.relative_to(online.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_files},
        "ranks": [{"rank": rank, "prospective_workspace_bytes": value}
                  for rank, value in floors.items()], "proof_tasks": records}
    canonical = _canonical(payload)
    return ManifestWorkspaceCertificate(_sha(canonical), hardware_fingerprint,
        floors, canonical, CAPACITY_REVISION, COVERAGE_REVISION, train_max_iter,
        manifest_sha, config_sha, context.config_hash)
