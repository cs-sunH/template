#!/usr/bin/env python3
"""Pure FACE request-mapping model shared by the online scheduling path.

The FACE paper describes a live host scheduler.  This module captures its
deterministic queue ordering, schedulable-instance range, direct cost
estimates, and per-die incremental decode cost without depending on
protobuf or ASTRA-sim internals.  The online scheduler reuses these
semantics while timing and progress come from the running simulator.

Cost source (stage D, 2026-10-03, design doc §8.4): the local-compute cost
of an iteration comes either from the legacy analytical roofline (default,
backward-compatible regression path) or from the die-internal cost library
through the ctypes oracle ``online/die_model_oracle.py`` (``cost_oracle=``
argument).  Only the local compute time source is swapped; the scoring
combination, candidate set, constraints, tie-break and KV rules are
unchanged.  No time or derivation formula of the die model is reimplemented
here -- every die number comes from ``die_estimate_batch`` over the wire.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, replace
from typing import Optional, Sequence

from online.die_model_oracle import DieModelOracle
from session_kv_manager import (
    estimate_model_weight_bytes,
)

@dataclass(frozen=True)
class FaceHardware:
    mesh_rows: int
    mesh_cols: int
    local_hbm_capacity_bytes: int
    # Unit convention (A1-17, 2026-09-26): despite the legacy "gbps" suffix,
    # both bandwidth fields carry giga-BYTES per second -- the value the
    # hardware JSON reports as "bandwidth-gbps" (its own notes call it
    # "GB/s").  Consumers multiply by 1e9 to obtain bytes/s;
    # schedulable_distance_limit below is a unit-less ratio of the two.
    local_hbm_bandwidth_gbps: float
    d2d_bandwidth_gbps: float
    peak_perf_tflops: float
    d2d_latency_ns: int
    local_hbm_latency_ns: int
    label: str = "FACE"

    def __post_init__(self) -> None:
        integer_fields = {
            "mesh_rows": self.mesh_rows,
            "mesh_cols": self.mesh_cols,
            "local_hbm_capacity_bytes": self.local_hbm_capacity_bytes,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in {
            "d2d_latency_ns": self.d2d_latency_ns,
            "local_hbm_latency_ns": self.local_hbm_latency_ns,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name, value in {
            "local_hbm_bandwidth_gbps": self.local_hbm_bandwidth_gbps,
            "d2d_bandwidth_gbps": self.d2d_bandwidth_gbps,
            "peak_perf_tflops": self.peak_perf_tflops,
        }.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def npus_count(self) -> int:
        return self.mesh_rows * self.mesh_cols

    @property
    def schedulable_distance_limit(self) -> float:
        return self.d2d_bandwidth_gbps / self.local_hbm_bandwidth_gbps


def coarse_die_face_peak_scalars(summary: dict) -> tuple[float, float]:
    """Convert typed library peaks to FACE selector units without deriving costs."""
    derived = summary.get("derived", summary.get("derived_hardware"))
    typed = derived.get("hardware_summary", {}) if isinstance(derived, dict) else {}
    row_rates = typed.get("hbm_row_peak_Bps")
    peak_int4 = typed.get("peak_int4_ops_per_s")
    if (not isinstance(row_rates, list) or len(row_rates) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value <= 0
                   for value in [peak_int4, *row_rates])):
        raise RuntimeError(
            "coarse_die_v2 hardware summary is missing positive finite typed "
            "compute/HBM peaks")
    return float(peak_int4) / 1e12, sum(map(float, row_rates)) / 1e9


def die_derived_face_hardware(
    hardware: FaceHardware, cost_oracle: DieModelOracle
) -> FaceHardware:
    """backend=die 时的 FaceHardware 标量注入(阶段 D;构造点在
    FaceOnlineScheduler/online_service 注入 oracle)。

    peak_perf_tflops / local_hbm_bandwidth_gbps 改由 oracle.hardware_summary()
    的 derived 块提供(legacy兼容标量回流；PE字段为废弃的INT4 TOPS兼容值，
    不表示BF16/FP32实测吞吐)。mesh 几何、HBM 容量
    (capacity-profile 口径,契约 §3.4)、D2D 带宽/时延与 local HBM 时延
    (契约 §3.4 沿用 hardware JSON)保持调用方传入值。backend=legacy 的
    调用方不经过本函数(回归保护:仍从 hardware JSON 读)。
    """
    derived = cost_oracle.hardware_summary().get("derived")
    if cost_oracle.coarse:
        peak_perf, hbm_bandwidth = coarse_die_face_peak_scalars(
            cost_oracle.hardware_summary())
        return replace(hardware, peak_perf_tflops=peak_perf,
                       local_hbm_bandwidth_gbps=hbm_bandwidth)
    if not derived or "pe_peak_tflops" not in derived \
            or "hbm_bandwidth_bytes" not in derived:
        raise RuntimeError(
            "die hardware summary is missing the derived scalars "
            "(pe_peak_tflops/hbm_bandwidth_bytes)")
    return replace(
        hardware,
        peak_perf_tflops=float(derived["pe_peak_tflops"]),
        local_hbm_bandwidth_gbps=float(derived["hbm_bandwidth_bytes"]) / 1e9,
    )


@dataclass(frozen=True)
class FaceModel:
    layers: int
    hidden_size: int
    ffn_size: int
    num_heads: int
    vocab_size: int
    bytes_per_elem: int
    mlp_variant: str = "gelu"

    def __post_init__(self) -> None:
        for name in (
            "layers",
            "hidden_size",
            "ffn_size",
            "num_heads",
            "vocab_size",
            "bytes_per_elem",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.mlp_variant not in {"gelu", "swiglu"}:
            raise ValueError("mlp_variant must be gelu or swiglu")


@dataclass(frozen=True)
class FaceInstanceSpec:
    name: str
    pg_name: str
    ranks: tuple[int, ...]


@dataclass(frozen=True)
class FaceInstance:
    index: int
    name: str
    pg_name: str
    ranks: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.ranks)


@dataclass(frozen=True)
class FaceTopology:
    hardware: FaceHardware
    instances: tuple[FaceInstance, ...]
    adjacency: tuple[tuple[int, ...], ...]

    def instance(self, index: int) -> FaceInstance:
        return self.instances[index]


def rank_coordinates(hardware: FaceHardware, rank: int) -> tuple[int, int]:
    if rank < 0 or rank >= hardware.npus_count:
        raise ValueError(f"rank {rank} is outside 0-{hardware.npus_count - 1}")
    return divmod(rank, hardware.mesh_cols)


def build_instances(
    hardware: FaceHardware,
    specs: Sequence[FaceInstanceSpec],
    *,
    require_equal_size: bool = True,
) -> FaceTopology:
    if not specs:
        raise ValueError("at least one FACE instance is required")

    all_ranks: list[int] = []
    instances: list[FaceInstance] = []
    rank_owner: dict[int, int] = {}
    pg_names: set[str] = set()

    for index, spec in enumerate(specs):
        if not spec.name:
            raise ValueError("instance name must not be empty")
        if not spec.pg_name or spec.pg_name == "0":
            raise ValueError("instance pg_name must be non-zero")
        if spec.pg_name in pg_names:
            raise ValueError(f"duplicate pg_name: {spec.pg_name}")
        pg_names.add(spec.pg_name)
        if not spec.ranks:
            raise ValueError(f"instance {spec.name} has no ranks")
        if len(set(spec.ranks)) != len(spec.ranks):
            raise ValueError(f"instance {spec.name} contains duplicate ranks")

        coordinates: set[tuple[int, int]] = set()
        for rank in spec.ranks:
            if rank in rank_owner:
                raise ValueError(
                    f"rank {rank} belongs to both instance {rank_owner[rank]} and {index}"
                )
            row, col = rank_coordinates(hardware, rank)
            coordinates.add((row, col))
            rank_owner[rank] = index
            all_ranks.append(rank)

        rows = [row for row, _ in coordinates]
        cols = [col for _, col in coordinates]
        row_min, row_max = min(rows), max(rows)
        col_min, col_max = min(cols), max(cols)
        expected = {
            (row, col)
            for row in range(row_min, row_max + 1)
            for col in range(col_min, col_max + 1)
        }
        if coordinates != expected:
            raise ValueError(
                f"instance {spec.name} must be a filled axis-aligned rectangle"
            )
        instances.append(
            FaceInstance(
                index=index,
                name=spec.name,
                pg_name=spec.pg_name,
                ranks=tuple(spec.ranks),
            )
        )

    if sorted(all_ranks) != list(range(hardware.npus_count)):
        raise ValueError("FACE instances must cover every configured NPU exactly once")
    if require_equal_size and len({instance.size for instance in instances}) != 1:
        raise ValueError(
            "direct KV shard pairing requires equal-size instances"
        )

    adjacency_sets = [set() for _ in instances]
    for rank in range(hardware.npus_count):
        row, col = rank_coordinates(hardware, rank)
        owner = rank_owner[rank]
        for drow, dcol in ((0, 1), (1, 0)):
            other_row, other_col = row + drow, col + dcol
            if other_row >= hardware.mesh_rows or other_col >= hardware.mesh_cols:
                continue
            other_rank = other_row * hardware.mesh_cols + other_col
            other_owner = rank_owner[other_rank]
            if owner != other_owner:
                adjacency_sets[owner].add(other_owner)
                adjacency_sets[other_owner].add(owner)

    if len(instances) > 1 and any(not neighbors for neighbors in adjacency_sets):
        raise ValueError("instance tiling must form a connected adjacency graph")

    return FaceTopology(
        hardware=hardware,
        instances=tuple(instances),
        adjacency=tuple(tuple(sorted(neighbors)) for neighbors in adjacency_sets),
    )


# estimate_model_weight_bytes lives in session_kv_manager (canonical copy,
# A1-15 2026-09-26) and is imported above; the behavior is unchanged.


def kv_cache_bytes_for_tokens(model: FaceModel, tokens: int) -> int:
    if tokens < 0:
        raise ValueError("KV token count must be non-negative")
    return 2 * model.layers * tokens * model.hidden_size * model.bytes_per_elem


@dataclass(frozen=True)
class FaceRooflineEstimate:
    instance_size: int
    p_chunk: int
    d_batch: int
    d_token: int
    iteration_time_ns: int
    source: str = "die_model_isolated"


# 估值来源标签(阶段 D):die 路径 = die_model_isolated(孤立估值);
# legacy 路径显式带 analytical_roofline 身份(设计文档 §8.4:凡参与策略
# 代价的本地计算项换用同源估值,旧 roofline 字段保留明确 legacy 身份)。
ESTIMATE_SOURCE_DIE = "die_model_isolated"
ESTIMATE_SOURCE_LEGACY = "analytical_roofline"


def _shard_extent(value: int, shards: int, shard_index: int) -> int:
    """与 generate_trace.shard_extent 同义的 TP 分片规则(前
    ``value % shards`` 个 rank 多拿一个元素,§4.2 规则 5"非整除 TP 用
    现有 shard_extent() 分片")。本模块刻意不 import generate_trace
    (保持零 protobuf/ASTRA-sim 依赖,见模块 docstring),故留本地同构
    副本;rank 0 恒为最大分片 = TP 组 straggler 代表。"""
    if value <= 0 or shards <= 0 or not 0 <= shard_index < shards:
        raise ValueError("invalid uneven shard specification")
    return value // shards + (1 if shard_index < value % shards else 0)


def _die_model_dims(model: FaceModel) -> dict:
    """FaceModel -> 契约 §2 model_dims。num_kv_heads == num_q_heads:
    本仓模型只有单一头数,按原有 MHA 解释(§4.2 规则 6)。norm_type/
    position_encoding 是 llama2_7b 模型族事实(FaceModel 不携带这两项),
    与 generate_trace 的算子域一致。"""
    bytes_per_elem = model.bytes_per_elem
    if bytes_per_elem not in (2, 4):
        raise ValueError(
            "die model dtype maps bytes_per_elem 2->bf16 / 4->fp32, got "
            "{}".format(bytes_per_elem))
    return {
        "hidden": model.hidden_size,
        "ffn": model.ffn_size,
        "vocab": model.vocab_size,
        "mlp_variant": model.mlp_variant,
        "norm_type": "rmsnorm",
        "position_encoding": "rope",
        "num_q_heads": model.num_heads,
        "num_kv_heads": model.num_heads,
        "head_dim": model.hidden_size // model.num_heads,
    }


def _die_tp_slice(model: FaceModel, instance_size: int) -> dict:
    """契约 §2 tp_slice:instance = 一个 TP 组,spec 描述组内单个 die
    (NPU)的本分片孤立执行;跨 die 部分和归原有 TP collective(§4.2
    规则 4),不进本估值。rank 0 = 最大分片(straggler 代表)。"""
    return {
        "tensor_parallel": instance_size,
        "tensor_parallel_rank": 0,
        "q_heads_local": _shard_extent(model.num_heads, instance_size, 0),
        "kv_heads_local": _shard_extent(model.num_heads, instance_size, 0),
        "ffn_local": _shard_extent(model.ffn_size, instance_size, 0),
        "vocab_local": _shard_extent(model.vocab_size, instance_size, 0),
    }


def build_die_estimate_specs(
    model: FaceModel,
    *,
    instance_size: int,
    p_chunk: int,
    d_batch: int,
    d_token: int,
    config_hash: str,
) -> tuple[list[dict], tuple[str, ...]]:
    """构造 estimate_iteration_time_ns die 路径的 spec 集(一次
    die_estimate_batch 批量的全部 spec,契约 §2 逐字段合规)。

    算子域(对齐旧 roofline 公式的线性/两段 attention 域,逐项对照表
    见 stage D 报告):
      - linear:M = p_chunk + d_batch 合并批(§4.2 规则 1),五个聚合
        规格 norm / qkv_rope / o_proj / ffn_region / residual,各
        layer_count = layers、weight_passes = 1(权重每层一次的现行
        scheduler 口径);
      - prefill attention_region:spans = [(p_chunk, kv=p_chunk,
        绝对位置 0/0)];
      - decode attention_region:spans = d_batch 个 (1, d_token)。

    返回 (specs, labels);labels[i] 是 specs[i] 的语义名,供调用方把
    批量结果分桶回 linear/prefill/decode 三段。
    """
    if instance_size <= 0 or p_chunk < 0 or d_batch < 0 or d_token < 0:
        raise ValueError("invalid die estimate workload parameters")
    if d_batch and d_token < 1:
        # decode span 的 kv_tokens >= 1 是契约 §2 硬约束;生产路径
        # current_decode_token 恒 >= 1(manifest 校验),此处早期拒绝
        # 非法直连调用,给出比库侧字段错误更明确的语义报错。
        raise ValueError(
            "die decode estimate requires d_token >= 1 when d_batch > 0")
    dims = _die_model_dims(model)
    tp_slice = _die_tp_slice(model, instance_size)
    dtype = "bf16" if model.bytes_per_elem == 2 else "fp32"
    phase = "prefill" if p_chunk else "decode"
    provenance = "face_scheduler.estimate_iteration_time_ns"

    def span(request_id: str, q_tokens: int, kv_tokens: int,
             q_start_pos: int, kv_start_pos: int) -> dict:
        return {
            "request_id": request_id,
            "q_tokens": q_tokens,
            "kv_tokens": kv_tokens,
            "q_start_pos": q_start_pos,
            "kv_start_pos": kv_start_pos,
        }

    # 同一次物理 batch 的片段集合:1 个 prefill chunk + d_batch 个 decode
    # 请求各 1 个新 token。linear 规格只用 sum(q_tokens)(=M);decode
    # 片段的绝对位置 = 其 KV 末位(q_start_pos = d_token - 1)。
    prefill_span = span("prefill_chunk", p_chunk, p_chunk, 0, 0) if p_chunk else None
    decode_spans = [
        span("decode_{}".format(index), 1, d_token, d_token - 1, 0)
        for index in range(d_batch)
    ]
    batch_spans = (
        ([prefill_span] if prefill_span is not None else []) + decode_spans
    )
    if not batch_spans:
        raise ValueError(
            "die estimate requires p_chunk > 0 or d_batch > 0 "
            "(idle workloads are short-circuited by the caller)")

    def spec(op_kind: str, spec_phase: str, spans: list[dict]) -> dict:
        return {
            "schema_version": 1,
            "op_kind": op_kind,
            "phase": spec_phase,
            "dtype": dtype,
            "model_dims": dims,
            "tp_slice": tp_slice,
            "spans": spans,
            "layer_count": model.layers,
            "weight_passes": 1,
            "kv_access": {"identity": "local"},
            "fusion_policy": "auto",
            "provenance": provenance,
            "config_hash": config_hash,
        }

    specs: list[dict] = []
    labels: list[str] = []
    for op_kind in ("norm", "qkv_rope", "o_proj", "ffn_region", "residual"):
        specs.append(spec(op_kind, phase, batch_spans))
        labels.append(op_kind)
    if prefill_span is not None:
        specs.append(spec("attention_region", "prefill", [prefill_span]))
        labels.append("attention_prefill")
    if decode_spans:
        specs.append(spec("attention_region", "decode", decode_spans))
        labels.append("attention_decode")
    return specs, tuple(labels)


def _estimate_iteration_time_ns_die(
    hardware: FaceHardware,
    model: FaceModel,
    cost_oracle: DieModelOracle,
    *,
    instance_size: int,
    p_chunk: int,
    d_batch: int,
    d_token: int,
) -> int:
    """die 路径:本地计算时间 = oracle 孤立估值;组合结构与 legacy 路径
    逐项相同(linear + max(prefill_attn, decode_attn) + d2d_latency_ns、
    max(1, ceil(...)) 取整)。全部数值来自 die_estimate_batch,本函数
    不实现任何时间公式;新 coarse C ABI 可用时只读取 cost-only 结果。"""
    specs, labels = build_die_estimate_specs(
        model,
        instance_size=instance_size,
        p_chunk=p_chunk,
        d_batch=d_batch,
        d_token=d_token,
        config_hash=cost_oracle.config_hash,
    )
    cost_only = getattr(cost_oracle, "estimate_cost", None)
    use_cost_only = (
        getattr(cost_oracle, "coarse", False) and
        getattr(cost_oracle, "supports_estimate_cost", False) and
        callable(cost_only))
    estimate_endpoint = ("die_coarse_estimate_cost_batch_v2"
                         if use_cost_only else "die_estimate_batch")
    results = cost_only(specs) if use_cost_only else cost_oracle.estimate(specs)
    seconds: dict[str, float] = {}
    for label, result, error in zip(labels, results, cost_oracle.last_errors):
        if result is None:
            raise RuntimeError(
                "die estimate spec {!r} rejected by {}: {}".format(
                    label, estimate_endpoint, error))
        seconds[label] = float(result["total_seconds"])
    linear_seconds = sum(
        seconds[label]
        for label in ("norm", "qkv_rope", "o_proj", "ffn_region", "residual")
    )
    prefill_seconds = seconds.get("attention_prefill", 0.0)
    decode_seconds = seconds.get("attention_decode", 0.0)
    # FACE 使两段 attention 重叠;linear 算子是共享批处理部分(与 legacy
    # 路径同一组合结构)。d2d_latency_ns 无条件附加(逐跳 NoC 时延);
    # 配 ceil-to-1ns 下限保持非空估值为正且确定。die 路径不再单加
    # local_hbm_latency 项——内存时延语义已包含在库的阶段模型/运行时
    # 内存协议中(契约 §7 裁定 4),预测器不重复收费。
    total_seconds = linear_seconds + max(prefill_seconds, decode_seconds)
    return max(1, math.ceil(total_seconds * 1e9) + hardware.d2d_latency_ns)


def estimate_iteration_time_ns(
    hardware: FaceHardware,
    model: FaceModel,
    *,
    instance_size: int,
    p_chunk: int,
    d_batch: int,
    d_token: int,
    cost_oracle: Optional[DieModelOracle] = None,
) -> int:
    if instance_size <= 0 or p_chunk < 0 or d_batch < 0 or d_token < 0:
        raise ValueError("invalid Roofline workload parameters")
    # (2026-09-29) 原 p_context_tokens 生产死参已删:全仓无生产调用方传入
    # (默认 p_context_tokens = p_chunk 恒成立),仅单测传参。attention 口径
    # 相应取 max(1, p_chunk),与改前默认路径逐字节一致。
    if p_chunk == 0 and d_batch == 0:
        return 0

    if cost_oracle is not None:
        return _estimate_iteration_time_ns_die(
            hardware,
            model,
            cost_oracle,
            instance_size=instance_size,
            p_chunk=p_chunk,
            d_batch=d_batch,
            d_token=d_token,
        )

    h = model.hidden_size
    ffn = model.ffn_size
    layers = model.layers
    bytes_per_elem = model.bytes_per_elem
    aggregate_perf = instance_size * hardware.peak_perf_tflops * 1e12
    aggregate_bw = instance_size * hardware.local_hbm_bandwidth_gbps * 1e9
    local_hbm_latency_seconds = hardware.local_hbm_latency_ns / 1e9

    linear_tokens = p_chunk + d_batch
    mlp_ops_per_token = 6 * h * ffn if model.mlp_variant == "swiglu" else 4 * h * ffn
    linear_ops_per_token = layers * (8 * h * h + mlp_ops_per_token)
    linear_ops = linear_tokens * linear_ops_per_token
    weight_bytes = estimate_model_weight_bytes(model)
    activation_bytes = layers * linear_tokens * h * bytes_per_elem * 8
    linear_seconds = max(
        linear_ops / aggregate_perf,
        local_hbm_latency_seconds +
        (weight_bytes + activation_bytes) / aggregate_bw,
    )

    prefill_attn_ops = layers * 4 * p_chunk * max(1, p_chunk) * h
    prefill_attn_bytes = (
        layers * p_chunk * max(1, p_chunk) * bytes_per_elem * 4
    )
    prefill_seconds = 0.0
    if p_chunk:
        prefill_seconds = max(
            prefill_attn_ops / aggregate_perf,
            local_hbm_latency_seconds + prefill_attn_bytes / aggregate_bw,
        )

    decode_attn_ops = layers * 4 * d_batch * d_token * h
    decode_attn_bytes = layers * 2 * d_batch * d_token * h * bytes_per_elem
    decode_seconds = 0.0
    if d_batch:
        decode_seconds = max(
            decode_attn_ops / aggregate_perf,
            local_hbm_latency_seconds + decode_attn_bytes / aggregate_bw,
        )

    # FACE overlaps the two attention phases; linear operators remain a shared
    # batched portion.  The estimate also adds d2d_latency_ns unconditionally
    # (a per-hop NoC latency, not a conditional controller term); together
    # with the ceil-to-1ns floor this keeps non-idle estimates positive and
    # deterministic.
    total_seconds = linear_seconds + max(prefill_seconds, decode_seconds)
    return max(1, math.ceil(total_seconds * 1e9) + hardware.d2d_latency_ns)


@dataclass(frozen=True)
class PrefillQueueSnapshot:
    instance_index: int
    remaining_chunks: int
    last_arrival_ns: Optional[int]

    @property
    def ordering_key(self) -> tuple[int, int, int]:
        return (
            self.remaining_chunks,
            -1 if self.last_arrival_ns is None else self.last_arrival_ns,
            self.instance_index,
        )


def select_prefill_instance(queues: Sequence[PrefillQueueSnapshot]) -> int:
    if not queues:
        raise ValueError("at least one prefill queue is required")
    return min(queues, key=lambda queue: queue.ordering_key).instance_index


class WeightedInstanceGraph:
    def __init__(self, topology: FaceTopology) -> None:
        self.topology = topology
        self.weights: dict[tuple[int, int], int] = {}
        for source, neighbors in enumerate(topology.adjacency):
            for target in neighbors:
                self.weights[self._edge(source, target)] = 1

    @staticmethod
    def _edge(source: int, target: int) -> tuple[int, int]:
        return (source, target) if source < target else (target, source)

    def edge_weight(self, source: int, target: int) -> int:
        edge = self._edge(source, target)
        if edge not in self.weights:
            raise ValueError(f"instances {source} and {target} are not adjacent")
        return self.weights[edge]

    def shortest_path(self, source: int, target: int) -> tuple[float, tuple[int, ...]]:
        count = len(self.topology.instances)
        if source < 0 or source >= count or target < 0 or target >= count:
            raise ValueError("instance index is out of range")
        if source == target:
            return 0.0, (source,)

        best: dict[int, tuple[float, tuple[int, ...]]] = {source: (0.0, (source,))}
        heap: list[tuple[float, tuple[int, ...], int]] = [(0.0, (source,), source)]
        while heap:
            distance, path, node = heapq.heappop(heap)
            if best.get(node) != (distance, path):
                continue
            if node == target:
                return distance, path
            for neighbor in self.topology.adjacency[node]:
                next_distance = distance + self.edge_weight(node, neighbor)
                next_path = path + (neighbor,)
                previous = best.get(neighbor)
                if previous is None or (next_distance, next_path) < previous:
                    best[neighbor] = (next_distance, next_path)
                    heapq.heappush(heap, (next_distance, next_path, neighbor))
        raise ValueError(f"no instance path from {source} to {target}")

    def schedulable_instances(self, source: int, limit: float) -> tuple[tuple[int, float], ...]:
        candidates = []
        for instance in self.topology.instances:
            distance, _ = self.shortest_path(source, instance.index)
            if distance <= limit + 1e-12:
                candidates.append((instance.index, distance))
        return tuple(candidates)

    # increase_path/decrease_path and the KVAllocator cluster were removed on
    # 2026-09-26 (M5-3): the only writers of self.weights, they served the
    # deleted remote-memory/cross-instance KV placement path that contradicts
    # the two-state resident-or-evicted session model.  Edge weights are the
    # constant 1 set in __init__, so edge_weight/shortest_path above remain
    # the live read side.


@dataclass(frozen=True)
class DecodeCandidateCost:
    instance_index: int
    weighted_distance: float
    current_roofline: FaceRooflineEstimate
    updated_roofline: FaceRooflineEstimate
    delta_time_ns: int
    per_die_delta_ns: float


class DecodeTieCounter:
    """decode 平局轮流裁决计数器（中-1 裁决 2026-08-20）：仅当候选集中
    ≥2 个 per_die_delta_ns 精确相等（真实平局）时前进；tied 集按
    instance_index 升序，取 counter % len(tied)。同一决策序列下确定可
    重放；不传入计数器时保持旧"最小 instance_index"行为。"""

    def __init__(self) -> None:
        self.value = 0


def select_decode_instance(
    *,
    topology: FaceTopology,
    graph: WeightedInstanceGraph,
    hardware: FaceHardware,
    model: FaceModel,
    fixed_p_chunk: int,
    prefill_instance_index: int,
    has_prefill_work: Sequence[bool],
    decode_token_lengths: Sequence[Sequence[int]],
    new_request_token_length: int,
    tie_counter: Optional[DecodeTieCounter] = None,
    cost_oracle: Optional[DieModelOracle] = None,
) -> tuple[int, tuple[DecodeCandidateCost, ...]]:
    """平局规则（中-1 裁决 2026-08-20）：per_die_delta_ns 精确并列时按
    instance_index 升序轮流（传入 tie_counter 时）；否则选最小
    instance_index（旧行为）。

    cost_oracle（阶段 D,可选）:非 None 时 estimate_iteration_time_ns
    走 die 孤立估值源;候选集合/评分/tie-break 逻辑不变。"""
    if len(has_prefill_work) != len(topology.instances):
        raise ValueError("has_prefill_work length must match instances")
    if len(decode_token_lengths) != len(topology.instances):
        raise ValueError("decode_token_lengths length must match instances")
    candidates = graph.schedulable_instances(
        prefill_instance_index,
        topology.hardware.schedulable_distance_limit,
    )
    if not candidates:
        raise ValueError("decode schedulable instance map is empty")

    estimate_source = (
        ESTIMATE_SOURCE_DIE if cost_oracle is not None
        else ESTIMATE_SOURCE_LEGACY
    )
    costs: list[DecodeCandidateCost] = []
    for instance_index, distance in candidates:
        instance = topology.instance(instance_index)
        active_tokens = tuple(decode_token_lengths[instance_index])
        p_chunk = fixed_p_chunk if has_prefill_work[instance_index] else 0
        d_batch = len(active_tokens)
        d_token = max(active_tokens, default=0)
        current = FaceRooflineEstimate(
            instance_size=instance.size,
            p_chunk=p_chunk,
            d_batch=d_batch,
            d_token=d_token,
            iteration_time_ns=estimate_iteration_time_ns(
                hardware,
                model,
                instance_size=instance.size,
                p_chunk=p_chunk,
                d_batch=d_batch,
                d_token=d_token,
                cost_oracle=cost_oracle,
            ),
            source=estimate_source,
        )
        updated_d_token = max(d_token, new_request_token_length)
        updated = FaceRooflineEstimate(
            instance_size=instance.size,
            p_chunk=p_chunk,
            d_batch=d_batch + 1,
            d_token=updated_d_token,
            iteration_time_ns=estimate_iteration_time_ns(
                hardware,
                model,
                instance_size=instance.size,
                p_chunk=p_chunk,
                d_batch=d_batch + 1,
                d_token=updated_d_token,
                cost_oracle=cost_oracle,
            ),
            source=estimate_source,
        )
        delta = updated.iteration_time_ns - current.iteration_time_ns
        costs.append(
            DecodeCandidateCost(
                instance_index=instance_index,
                weighted_distance=distance,
                current_roofline=current,
                updated_roofline=updated,
                delta_time_ns=delta,
                per_die_delta_ns=delta / instance.size,
            )
        )
    min_delta = min(cost.per_die_delta_ns for cost in costs)
    tied = sorted(
        (cost for cost in costs if cost.per_die_delta_ns == min_delta),
        key=lambda cost: cost.instance_index,
    )
    if len(tied) > 1 and tie_counter is not None:
        selected = tied[tie_counter.value % len(tied)]
        tie_counter.value += 1
    else:
        selected = tied[0]
    return selected.instance_index, tuple(costs)


