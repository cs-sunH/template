"""Immutable algebraic remote-read pipeline descriptors (wire v1).

This module is the Python counterpart of C++ ``PipelineDescriptor``.  It keeps
only bounded group/shard/selector data; tiles are derived on demand with checked
uint64 arithmetic.  Mutable execution progress belongs in PipelineCursor on the
C++ side and is never stored here.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping as _RuntimeMapping
import math
from typing import Any, Mapping, Optional, Sequence, Tuple


U64_MAX = (1 << 64) - 1
U32_MAX = (1 << 32) - 1
I32_MAX = (1 << 31) - 1
I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1

TARGET_RANK_ORDER = (
    "release_ack_send_i_if_required_then_data_recv_i_then_"
    "fill_ack_send_i_then_data_recv_i_plus_1")
SOURCE_RANK_ORDER = (
    "release_ack_recv_i_if_required_then_data_send_i_then_fill_ack_recv_i")
TILE_BRANCHING = "tile1_head_main_chain_tile2_immediate_side_branch_tile3_plus_release_gated"


class PipelineDescriptorError(ValueError):
    """Malformed, inconsistent, or out-of-range pipeline descriptor."""


def _mapping(value: Any, path: str, required: Sequence[str],
             optional: Sequence[str] = ()) -> Mapping[str, Any]:
    if not issubclass(type(value), _RuntimeMapping):
        raise PipelineDescriptorError(f"{path} must be an object")
    missing = set(required) - set(value)
    unknown = set(value) - set(required) - set(optional)
    if missing or unknown:
        raise PipelineDescriptorError(
            f"{path} keys invalid (missing={sorted(missing)}, "
            f"unknown={sorted(unknown)})")
    return value


def _uint(value: Any, path: str, maximum: int = U64_MAX) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise PipelineDescriptorError(
            f"{path} must be an integer in [0,{maximum}] (bool is invalid)")
    return value


def _positive(value: Any, path: str, maximum: int = U64_MAX) -> int:
    result = _uint(value, path, maximum)
    if result == 0:
        raise PipelineDescriptorError(f"{path} must be positive")
    return result


def _int64(value: Any, path: str) -> int:
    if type(value) is not int or not I64_MIN <= value <= I64_MAX:
        raise PipelineDescriptorError(
            f"{path} must be a signed 64-bit integer (bool is invalid)")
    return value


def _rank(value: Any, path: str) -> int:
    return _uint(value, path, I32_MAX)


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise PipelineDescriptorError(f"{path} must be a non-empty string")
    return value


def _enum(value: Any, path: str, allowed: Sequence[str]) -> str:
    if value not in allowed:
        raise PipelineDescriptorError(
            f"{path} must be one of {tuple(allowed)!r}, got {value!r}")
    return value


def _checked_add(a: int, b: int, path: str = "addition") -> int:
    if a < 0 or b < 0 or a > U64_MAX - b:
        raise PipelineDescriptorError(f"uint64 overflow in {path}")
    return a + b


def _checked_mul(a: int, b: int, path: str = "multiplication") -> int:
    if a < 0 or b < 0 or (a and b > U64_MAX // a):
        raise PipelineDescriptorError(f"uint64 overflow in {path}")
    return a * b


def _ceil_div(a: int, b: int) -> int:
    return a // b + (1 if a % b else 0)


def _array(value: Any, path: str, *, nonempty: bool = False) -> tuple:
    if not isinstance(value, (list, tuple)):
        raise PipelineDescriptorError(f"{path} must be an array")
    result = tuple(value)
    if nonempty and not result:
        raise PipelineDescriptorError(f"{path} must not be empty")
    return result


@dataclass(frozen=True)
class PipelineIdentity:
    request_id: str
    stage: str
    generation: int
    stage_epoch: int
    segment_ordinal: int
    rank: int
    staging_allocation_id: int
    segment_id: str
    buffer_domain_id: str


@dataclass(frozen=True)
class PipelineTileAxis:
    unit: str
    total_units: int
    units_per_tile: int
    tile_count: int
    tail_units: int
    global_tile_first: int


@dataclass(frozen=True)
class PipelineShardRule:
    shard_ordinal: int
    source_rank: int
    target_rank: int
    total_data_bytes: int
    bytes_per_unit: int
    noc_path: Tuple[int, ...]


@dataclass(frozen=True)
class PipelineGroup:
    group_ordinal: int
    pass_or_window_ordinal: int
    source_ordinal: int
    source_key: str
    layer_start: int
    layer_end: int
    tile_axis: PipelineTileAxis
    shards: Tuple[PipelineShardRule, ...]


@dataclass(frozen=True)
class PipelineProtocol:
    buffer_slots: int = 2
    release_lag_tiles: int = 2
    initial_tiles_without_release: int = 2
    fill_ack_bytes: int = 1
    release_ack_bytes: int = 1
    hbm_charge: bool = True
    target_rank_order: str = TARGET_RANK_ORDER
    source_rank_order: str = SOURCE_RANK_ORDER
    tile_branching: str = TILE_BRANCHING


@dataclass(frozen=True)
class PipelineDecodePredictionProfile:
    kind: str
    profile_id: str
    compute_ns_total: float
    hbm_bytes_total: int
    resource_key: str
    share_rule: str


@dataclass(frozen=True)
class PipelinePredictionProfile:
    prefill_consume: str
    decode_consume: PipelineDecodePredictionProfile


@dataclass(frozen=True)
class PipelineExecutionProfile:
    data_legs: str
    endpoint_hbm_rule: str
    prefill_consume: str
    decode_consume: str


@dataclass(frozen=True)
class PipelineLogicalIdRange:
    rank: int
    first_json_id: int
    count: int
    stride: int
    kind: str


@dataclass(frozen=True)
class PipelineTagAllocationRun:
    allocator_scope_id: str
    emission_ordinal: int
    first_tag: int
    count: int
    group_ordinal: int
    first_tile: int
    first_shard: int
    tile_count: int
    shard_count: int


@dataclass(frozen=True)
class PipelineEndpointLegRun:
    endpoint: str
    json_offset: int
    tag_emission_ordinal: Optional[int]
    tag_offset: Optional[int]
    tag_tile_stride: Optional[int]
    tag_shard_stride: Optional[int]


@dataclass(frozen=True)
class PipelineLegacyContext:
    request_id: str
    stage: str
    generation: int


@dataclass(frozen=True)
class PipelineEndpointRun:
    """A bounded affine mapping from (tile, shard, leg) to legacy endpoint IDs."""

    rank: int
    group_ordinal: int
    first_tile: int
    tile_count: int
    shard_ordinals: Tuple[int, ...]
    first_json_id: int
    tile_stride: int
    shard_stride: int
    initial_parent_json_id: Optional[int]
    legacy_context: PipelineLegacyContext
    legs: Tuple[PipelineEndpointLegRun, ...]


@dataclass(frozen=True)
class PipelineEndpointDependencyRun:
    """Affine ordinary/virtual parents for emitted endpoint positions."""

    rank: int
    group_ordinal: int
    first_tile: int
    tile_count: int
    shard_ordinals: Tuple[int, ...]
    endpoint: str
    dependency_kind: str
    first_dependency_json_id: int
    dependency_tile_stride: int
    dependency_shard_stride: int


@dataclass(frozen=True)
class PipelineConsumeCompletionRun:
    """Affine body/marker terminal that actually releases each tile's credit."""

    rank: int
    group_ordinal: int
    first_tile: int
    tile_count: int
    shard_ordinals: Tuple[int, ...]
    dependency_kind: str
    first_dependency_json_id: int
    dependency_tile_stride: int
    dependency_shard_stride: int


@dataclass(frozen=True)
class PipelineProjectionShare:
    rule: str
    total_units: int
    total_tiles: int
    share_index_first: int
    tile_bytes: int

    def at_global_index(self, index: int) -> int:
        """Return the exact legacy largest-remainder share at one index."""
        index = _uint(index, "projection share index")
        if index >= self.total_tiles:
            raise PipelineDescriptorError("projection share index is out of range")
        quotient, remainder = divmod(self.total_units, self.total_tiles)
        return quotient + (1 if index < remainder else 0)


@dataclass(frozen=True)
class PipelineProjectionFill:
    leg_id_prefix: str
    first_index: int
    fill_order_first: int
    fill_order_stride: int
    completion_order_first: Optional[int]
    completion_order_stride: int
    route: Tuple[str, ...]
    bytes_per_tile: int
    latency_ns: int
    start_deps: Tuple[str, ...]
    gate_id: str
    gate_capacity: int


@dataclass(frozen=True)
class PipelineProjectionConsume:
    leg_id_prefix: str
    first_index: int
    consume_order_first: int
    consume_order_stride: int
    completion_order_first: Optional[int]
    completion_order_stride: int
    kind: str
    route: Tuple[str, ...]
    group: Optional[str]
    latency_first_ns: int
    latency_rest_ns: int
    consume_deps: Tuple[str, ...]
    consume_start_deps: Tuple[str, ...]
    gate_id: str
    compute_share: Optional[PipelineProjectionShare]
    hbm_share: Optional[PipelineProjectionShare]


@dataclass(frozen=True)
class PipelineProjectionChain:
    group_ordinal: int
    rank: int
    tile_count: int
    fill: PipelineProjectionFill
    consume: PipelineProjectionConsume


@dataclass(frozen=True)
class PipelineProjection:
    version: int
    chains: Tuple[PipelineProjectionChain, ...]


@dataclass(frozen=True)
class PipelineAnchorSelector:
    kind: str
    group_ordinal: Optional[int] = None
    tile_ordinal: Optional[int] = None
    shard_ordinal: Optional[int] = None
    endpoint: Optional[str] = None
    existing_gate: Optional[str] = None
    tag_allocation_emission_ordinal: Optional[int] = None
    tag_offset: Optional[int] = None


@dataclass(frozen=True)
class PipelineAnchor:
    role: str
    rank: int
    binding: str
    node_id: Optional[int]
    selector: PipelineAnchorSelector


@dataclass(frozen=True)
class PipelineShardTile:
    shard_ordinal: int
    source_rank: int
    target_rank: int
    data_bytes: int
    emit_endpoint: bool


@dataclass(frozen=True)
class PipelineTile:
    global_tile_ordinal: int
    units: int
    data_bytes_total: int
    shards: Tuple[PipelineShardTile, ...]


def _tile_shard_data_bytes(axis: PipelineTileAxis, units: int,
                           offset: int, shard: PipelineShardRule) -> int:
    if axis.unit == "source_tokens":
        return _checked_mul(units, shard.bytes_per_unit,
                            "source-token shard bytes")
    return max(0, min(
        shard.total_data_bytes,
        _checked_add(offset, units, "stream-byte tile end")) - offset)


def _make_pipeline_shard_tile(
    shard: PipelineShardRule, data_bytes: int,
) -> PipelineShardTile:
    return PipelineShardTile(
        shard.shard_ordinal, shard.source_rank, shard.target_rank,
        data_bytes, data_bytes > 0)


@dataclass(frozen=True)
class PipelineDescriptor:
    """Fully validated immutable descriptor with lazy tile algebra."""

    identity: PipelineIdentity
    stream_kind: str
    groups: Tuple[PipelineGroup, ...]
    protocol: PipelineProtocol
    prediction_profile: PipelinePredictionProfile
    execution_profile: PipelineExecutionProfile
    logical_id_ranges: Tuple[PipelineLogicalIdRange, ...]
    tag_allocation_runs: Tuple[PipelineTagAllocationRun, ...]
    endpoint_runs: Tuple[PipelineEndpointRun, ...]
    endpoint_dependency_runs: Tuple[PipelineEndpointDependencyRun, ...]
    consume_completion_runs: Tuple[PipelineConsumeCompletionRun, ...]
    projection: PipelineProjection
    anchors: Tuple[PipelineAnchor, ...]
    descriptor_version: int = 1

    @classmethod
    def from_plan(cls, plan: Mapping[str, Any], **overrides) -> "PipelineDescriptor":
        """Validate an algebraic plan/wire mapping and freeze all nested data.

        The producers pass bounded group/range data.  Overrides support a small
        producer convenience surface while retaining one canonical schema.
        """
        if not issubclass(type(plan), _RuntimeMapping):
            raise PipelineDescriptorError("plan must be a mapping")
        if overrides:
            plan = {**plan, **overrides}
        return cls.from_wire(plan)

    @classmethod
    def from_wire(cls, wire: Mapping[str, Any]) -> "PipelineDescriptor":
        top = _mapping(wire, "descriptor", (
            "descriptor_version", "identity", "stream_kind", "groups",
            "protocol", "prediction_profile", "execution_profile",
            "logical_id_ranges", "tag_allocation_runs", "endpoint_runs",
            "endpoint_dependency_runs", "consume_completion_runs",
            "projection", "anchors"))
        version = _uint(top["descriptor_version"], "descriptor_version", 0xFFFFFFFF)
        if version != 1:
            raise PipelineDescriptorError(f"unsupported descriptor_version {version}")

        raw = _mapping(top["identity"], "identity", (
            "request_id", "stage", "generation", "stage_epoch",
            "segment_ordinal", "rank", "staging_allocation_id", "segment_id",
            "buffer_domain_id"))
        request = _string(raw["request_id"], "identity.request_id")
        stage = _enum(raw["stage"], "identity.stage", ("prefill", "decode"))
        generation = _uint(raw["generation"], "identity.generation")
        stage_epoch = _uint(raw["stage_epoch"], "identity.stage_epoch")
        segment_ordinal = _uint(raw["segment_ordinal"], "identity.segment_ordinal")
        rank = _rank(raw["rank"], "identity.rank")
        allocation = _uint(raw["staging_allocation_id"], "identity.staging_allocation_id")
        segment_id = _string(raw["segment_id"], "identity.segment_id")
        domain_id = _string(raw["buffer_domain_id"], "identity.buffer_domain_id")
        request_hex = request.encode("utf-8").hex()
        expected_segment = (
            f"pp1/{request_hex}/{stage}/{generation}/{stage_epoch}/"
            f"{rank}/{segment_ordinal}")
        expected_domain = f"pb1/{request_hex}/{rank}/{allocation}"
        if segment_id != expected_segment or domain_id != expected_domain:
            raise PipelineDescriptorError("identity IDs do not match canonical encoding")
        identity = PipelineIdentity(
            request, stage, generation, stage_epoch, segment_ordinal, rank,
            allocation, segment_id, domain_id)

        stream_kind = _enum(top["stream_kind"], "stream_kind", (
            "prefill_read_window", "decode_remote_credit"))
        groups = tuple(_parse_group(value, f"groups[{i}]")
                       for i, value in enumerate(_array(top["groups"], "groups", nonempty=True)))
        group_ids = tuple(group.group_ordinal for group in groups)
        if group_ids != tuple(sorted(set(group_ids))):
            raise PipelineDescriptorError("groups must preserve strictly increasing group_ordinal")
        expected_stage_kind = (
            "prefill_read_window" if stage == "prefill" else "decode_remote_credit")
        if stream_kind != expected_stage_kind:
            raise PipelineDescriptorError("stream_kind does not match identity.stage")

        protocol = _parse_protocol(top["protocol"])
        prediction = _parse_prediction_profile(top["prediction_profile"])
        execution = _parse_execution_profile(top["execution_profile"])
        id_ranges = tuple(_parse_id_range(value, f"logical_id_ranges[{i}]")
                          for i, value in enumerate(_array(top["logical_id_ranges"], "logical_id_ranges")))
        _validate_id_ranges(id_ranges)
        runs = tuple(_parse_tag_run(value, f"tag_allocation_runs[{i}]")
                     for i, value in enumerate(_array(top["tag_allocation_runs"], "tag_allocation_runs")))
        _validate_tag_runs(runs, groups)
        endpoint_runs = tuple(_parse_endpoint_run(value, f"endpoint_runs[{i}]")
                              for i, value in enumerate(_array(top["endpoint_runs"], "endpoint_runs")))
        _validate_endpoint_runs(endpoint_runs, groups, runs, id_ranges)
        endpoint_dependency_runs = tuple(
            _parse_endpoint_dependency_run(value,
                                           f"endpoint_dependency_runs[{i}]")
            for i, value in enumerate(_array(
                top["endpoint_dependency_runs"], "endpoint_dependency_runs")))
        consume_completion_runs = tuple(
            _parse_consume_completion_run(value,
                                          f"consume_completion_runs[{i}]")
            for i, value in enumerate(_array(
                top["consume_completion_runs"], "consume_completion_runs")))
        _validate_endpoint_dependency_runs(
            endpoint_dependency_runs, endpoint_runs, groups)
        if endpoint_runs:
            _validate_consume_completion_runs(
                consume_completion_runs, groups, identity.rank)
        elif consume_completion_runs:
            raise PipelineDescriptorError(
                "projection-only descriptor must not include consume completion runs")
        projection = _parse_projection(
            top["projection"], groups, protocol, identity.rank)
        anchors = tuple(_parse_anchor(value, f"anchors[{i}]")
                        for i, value in enumerate(_array(top["anchors"], "anchors")))
        execution_descriptor = bool(id_ranges or runs or endpoint_runs)
        if execution_descriptor:
            if not anchors:
                raise PipelineDescriptorError(
                    "execution descriptor needs endpoint/service anchors")
            _validate_anchors(anchors, groups, runs, endpoint_runs)
            _validate_endpoint_id_targets(endpoint_runs, id_ranges, anchors)
            roles = {anchor.role for anchor in anchors}
            if not {"start", "frontier", "service", "settle"}.issubset(roles):
                raise PipelineDescriptorError(
                    "execution descriptor needs start/frontier/service/settle anchors")
        elif anchors:
            raise PipelineDescriptorError(
                "projection-only descriptor must not invent execution anchors")
        return cls(identity, stream_kind, groups, protocol, prediction,
                   execution, id_ranges, runs, endpoint_runs,
                   endpoint_dependency_runs, consume_completion_runs,
                   projection, anchors, version)

    def to_wire(self) -> dict:
        """Return fresh JSON-compatible containers in canonical schema shape."""
        return {
            "descriptor_version": self.descriptor_version,
            "identity": {
                "request_id": self.identity.request_id,
                "stage": self.identity.stage,
                "generation": self.identity.generation,
                "stage_epoch": self.identity.stage_epoch,
                "segment_ordinal": self.identity.segment_ordinal,
                "rank": self.identity.rank,
                "staging_allocation_id": self.identity.staging_allocation_id,
                "segment_id": self.identity.segment_id,
                "buffer_domain_id": self.identity.buffer_domain_id,
            },
            "stream_kind": self.stream_kind,
            "groups": [_group_wire(group) for group in self.groups],
            "protocol": {
                "buffer_slots": self.protocol.buffer_slots,
                "release_lag_tiles": self.protocol.release_lag_tiles,
                "initial_tiles_without_release": self.protocol.initial_tiles_without_release,
                "fill_ack_bytes": self.protocol.fill_ack_bytes,
                "release_ack_bytes": self.protocol.release_ack_bytes,
                "hbm_charge": self.protocol.hbm_charge,
                "target_rank_order": self.protocol.target_rank_order,
                "source_rank_order": self.protocol.source_rank_order,
                "tile_branching": self.protocol.tile_branching,
            },
            "prediction_profile": {
                "prefill_consume": self.prediction_profile.prefill_consume,
                "decode_consume": {
                    "kind": self.prediction_profile.decode_consume.kind,
                    "profile_id": self.prediction_profile.decode_consume.profile_id,
                    "compute_ns_total": self.prediction_profile.decode_consume.compute_ns_total,
                    "hbm_bytes_total": self.prediction_profile.decode_consume.hbm_bytes_total,
                    "resource_key": self.prediction_profile.decode_consume.resource_key,
                    "share_rule": self.prediction_profile.decode_consume.share_rule,
                },
            },
            "execution_profile": {
                "data_legs": self.execution_profile.data_legs,
                "endpoint_hbm_rule": self.execution_profile.endpoint_hbm_rule,
                "prefill_consume": self.execution_profile.prefill_consume,
                "decode_consume": self.execution_profile.decode_consume,
            },
            "logical_id_ranges": [
                {"rank": item.rank, "first_json_id": item.first_json_id,
                 "count": item.count, "stride": item.stride, "kind": item.kind}
                for item in self.logical_id_ranges],
            "tag_allocation_runs": [
                {"allocator_scope_id": item.allocator_scope_id,
                 "emission_ordinal": item.emission_ordinal,
                 "first_tag": item.first_tag, "count": item.count,
                 "group_ordinal": item.group_ordinal,
                 "first_tile": item.first_tile, "first_shard": item.first_shard,
                 "tile_count": item.tile_count, "shard_count": item.shard_count}
                for item in self.tag_allocation_runs],
            "endpoint_runs": [
                _endpoint_run_wire(item) for item in self.endpoint_runs],
            "endpoint_dependency_runs": [
                _endpoint_dependency_run_wire(item)
                for item in self.endpoint_dependency_runs],
            "consume_completion_runs": [
                _consume_completion_run_wire(item)
                for item in self.consume_completion_runs],
            "projection": _projection_wire(self.projection),
            "anchors": [_anchor_wire(anchor) for anchor in self.anchors],
        }

    def _group(self, group_ordinal: int) -> PipelineGroup:
        group_ordinal = _uint(group_ordinal, "group_ordinal")
        for group in self.groups:
            if group.group_ordinal == group_ordinal:
                return group
        raise PipelineDescriptorError(f"unknown group_ordinal {group_ordinal}")

    def tile(self, group_ordinal: int, local_tile_ordinal: int) -> PipelineTile:
        """Derive one tile; local ordinal is zero-based, global is one-based."""
        group = self._group(group_ordinal)
        local = _uint(local_tile_ordinal, "local_tile_ordinal")
        axis = group.tile_axis
        if local >= axis.tile_count:
            raise PipelineDescriptorError(
                f"local tile {local} outside group {group.group_ordinal} count {axis.tile_count}")
        global_ordinal = _checked_add(axis.global_tile_first, local, "global tile ordinal")
        units = (axis.tail_units if local + 1 == axis.tile_count
                 else axis.units_per_tile)
        offset = _checked_mul(local, axis.units_per_tile, "tile unit offset")
        shards = []
        total_bytes = 0
        for shard in group.shards:
            data_bytes = _tile_shard_data_bytes(axis, units, offset, shard)
            total_bytes = _checked_add(total_bytes, data_bytes, "tile total bytes")
            shards.append(_make_pipeline_shard_tile(shard, data_bytes))
        return PipelineTile(global_ordinal, units, total_bytes, tuple(shards))

    def selected_shard_tile(
        self, group_ordinal: int, local_tile_ordinal: int,
        group_shard_index: int,
    ) -> Tuple[int, PipelineShardTile]:
        """Derive one selected shard tile without constructing its siblings.

        ``local_tile_ordinal`` and ``group_shard_index`` are zero-based. The
        returned global tile ordinal remains one-based, matching :meth:`tile`.
        """
        group = self._group(group_ordinal)
        local = _uint(local_tile_ordinal, "local_tile_ordinal")
        axis = group.tile_axis
        if local >= axis.tile_count:
            raise PipelineDescriptorError(
                f"local tile {local} outside group {group.group_ordinal} count {axis.tile_count}")
        shard_index = _uint(group_shard_index, "group_shard_index")
        if shard_index >= len(group.shards):
            raise PipelineDescriptorError(
                f"shard index {shard_index} outside group {group.group_ordinal} count {len(group.shards)}")

        global_ordinal = _checked_add(
            axis.global_tile_first, local, "global tile ordinal")
        units = (axis.tail_units if local + 1 == axis.tile_count
                 else axis.units_per_tile)
        offset = _checked_mul(local, axis.units_per_tile, "tile unit offset")
        shard = group.shards[shard_index]
        data_bytes = _tile_shard_data_bytes(axis, units, offset, shard)
        return global_ordinal, _make_pipeline_shard_tile(shard, data_bytes)

    def total(self, group_ordinal: int) -> int:
        """Sum exact group payload bytes without enumerating tiles."""
        group = self._group(group_ordinal)
        total = 0
        for shard in group.shards:
            total = _checked_add(total, shard.total_data_bytes, "group total bytes")
        return total

    def projection_terminal_ids(self) -> Tuple[str, ...]:
        """Return the bounded service frontier IDs for compact projection chains.

        Each consume chain is serial, so its final logical leg is the exact
        maximum for the chain.  The caller may combine these IDs with ordinary
        service legs without expanding the virtual tile range.
        """
        return tuple(
            "{}{}".format(
                chain.consume.leg_id_prefix,
                chain.consume.first_index + chain.tile_count - 1)
            for chain in self.projection.chains)

    def prefix(self, group_ordinal: int, count: int) -> int:
        """Sum the first ``count`` tile payloads using bounded algebra."""
        group = self._group(group_ordinal)
        count = _uint(count, "prefix count")
        axis = group.tile_axis
        if count > axis.tile_count:
            raise PipelineDescriptorError(
                f"prefix count {count} exceeds tile_count {axis.tile_count}")
        consumed_units = min(axis.total_units,
                             _checked_mul(count, axis.units_per_tile, "prefix units"))
        total = 0
        for shard in group.shards:
            if axis.unit == "source_tokens":
                value = _checked_mul(consumed_units, shard.bytes_per_unit,
                                     "source-token prefix bytes")
            else:
                value = min(consumed_units, shard.total_data_bytes)
            total = _checked_add(total, value, "prefix total bytes")
        return total

    def endpoint_id(self, run_ordinal: int, tile_ordinal: int,
                    shard_ordinal: int, leg_ordinal: int) -> int:
        """Resolve one endpoint's legacy logical ID without expanding its run."""
        run_ordinal = _uint(run_ordinal, "endpoint run ordinal")
        if run_ordinal >= len(self.endpoint_runs):
            raise PipelineDescriptorError("endpoint run ordinal is outside descriptor")
        run = self.endpoint_runs[run_ordinal]
        tile_ordinal = _uint(tile_ordinal, "endpoint tile ordinal")
        shard_ordinal = _uint(shard_ordinal, "endpoint shard ordinal")
        leg_ordinal = _uint(leg_ordinal, "endpoint leg ordinal")
        if not run.first_tile <= tile_ordinal < run.first_tile + run.tile_count:
            raise PipelineDescriptorError("tile is outside endpoint run")
        try:
            local_shard = run.shard_ordinals.index(shard_ordinal)
        except ValueError as exc:
            raise PipelineDescriptorError("shard is outside endpoint run") from exc
        if leg_ordinal >= len(run.legs):
            raise PipelineDescriptorError("leg is outside endpoint run")
        leg = run.legs[leg_ordinal]
        local_tile = tile_ordinal - run.first_tile
        value = _checked_add(run.first_json_id,
                             _checked_mul(local_tile, run.tile_stride, "endpoint tile ID offset"),
                             "endpoint ID")
        value = _checked_add(value,
                             _checked_mul(local_shard, run.shard_stride, "endpoint shard ID offset"),
                             "endpoint ID")
        return _checked_add(value, leg.json_offset, "endpoint leg ID")

    def endpoint_tag(self, run_ordinal: int, tile_ordinal: int,
                     shard_ordinal: int, leg_ordinal: int) -> Optional[int]:
        """Resolve an endpoint tag at its original allocator emission point."""
        run_ordinal = _uint(run_ordinal, "endpoint run ordinal")
        if run_ordinal >= len(self.endpoint_runs):
            raise PipelineDescriptorError("endpoint run ordinal is outside descriptor")
        run = self.endpoint_runs[run_ordinal]
        tile_ordinal = _uint(tile_ordinal, "endpoint tile ordinal")
        shard_ordinal = _uint(shard_ordinal, "endpoint shard ordinal")
        leg_ordinal = _uint(leg_ordinal, "endpoint leg ordinal")
        if not run.first_tile <= tile_ordinal < run.first_tile + run.tile_count:
            raise PipelineDescriptorError("tile is outside endpoint run")
        try:
            local_shard = run.shard_ordinals.index(shard_ordinal)
        except ValueError as exc:
            raise PipelineDescriptorError("shard is outside endpoint run") from exc
        if leg_ordinal >= len(run.legs):
            raise PipelineDescriptorError("leg is outside endpoint run")
        leg = run.legs[leg_ordinal]
        if leg.tag_emission_ordinal is None:
            return None
        tag_run = next((item for item in self.tag_allocation_runs
                        if item.emission_ordinal == leg.tag_emission_ordinal), None)
        if tag_run is None:
            raise PipelineDescriptorError("endpoint leg references missing tag run")
        local_tile = tile_ordinal - run.first_tile
        offset = _checked_add(leg.tag_offset,
                              _checked_mul(local_tile, leg.tag_tile_stride, "endpoint tile tag offset"),
                              "endpoint tag offset")
        offset = _checked_add(offset,
                              _checked_mul(local_shard, leg.tag_shard_stride, "endpoint shard tag offset"),
                              "endpoint tag offset")
        return _checked_add(tag_run.first_tag, offset, "endpoint tag")


def _parse_group(value: Any, path: str) -> PipelineGroup:
    raw = _mapping(value, path, ("group_ordinal", "pass_or_window_ordinal",
        "source_ordinal", "source_key", "layer_start", "layer_end", "tile_axis", "shards"))
    axis_raw = _mapping(raw["tile_axis"], path + ".tile_axis", (
        "unit", "total_units", "units_per_tile", "tile_count", "tail_units", "global_tile_first"))
    unit = _enum(axis_raw["unit"], path + ".tile_axis.unit", ("source_tokens", "stream_bytes"))
    total_units = _positive(axis_raw["total_units"], path + ".tile_axis.total_units")
    units_per_tile = _positive(axis_raw["units_per_tile"], path + ".tile_axis.units_per_tile")
    tile_count = _positive(axis_raw["tile_count"], path + ".tile_axis.tile_count")
    tail_units = _uint(axis_raw["tail_units"], path + ".tile_axis.tail_units")
    global_first = _positive(axis_raw["global_tile_first"], path + ".tile_axis.global_tile_first")
    expected_count = _ceil_div(total_units, units_per_tile)
    expected_tail = total_units - (expected_count - 1) * units_per_tile
    if tile_count != expected_count or tail_units != expected_tail:
        raise PipelineDescriptorError(f"{path}.tile_axis has inconsistent count/tail")
    _checked_add(global_first, tile_count - 1, "global tile range")
    axis = PipelineTileAxis(unit, total_units, units_per_tile, tile_count, tail_units, global_first)
    shards = []
    for i, item in enumerate(_array(raw["shards"], path + ".shards", nonempty=True)):
        shard_path = f"{path}.shards[{i}]"
        shard = _mapping(item, shard_path, ("shard_ordinal", "source_rank", "target_rank",
            "total_data_bytes", "bytes_per_unit", "noc_path"))
        ordinal = _uint(shard["shard_ordinal"], shard_path + ".shard_ordinal")
        if ordinal != i:
            raise PipelineDescriptorError(f"{shard_path}.shard_ordinal must be original zero-based order")
        source_rank = _rank(shard["source_rank"], shard_path + ".source_rank")
        target_rank = _rank(shard["target_rank"], shard_path + ".target_rank")
        total_bytes = _uint(shard["total_data_bytes"], shard_path + ".total_data_bytes")
        bpu = _uint(shard["bytes_per_unit"], shard_path + ".bytes_per_unit")
        path_value = tuple(_rank(r, f"{shard_path}.noc_path[{j}]")
                           for j, r in enumerate(_array(shard["noc_path"], shard_path + ".noc_path")))
        if path_value and (path_value[0] != source_rank or path_value[-1] != target_rank):
            raise PipelineDescriptorError(f"{shard_path}.noc_path must begin/end at shard ranks")
        if unit == "source_tokens":
            if bpu == 0 or _checked_mul(total_units, bpu, "shard total bytes") != total_bytes:
                raise PipelineDescriptorError(f"{shard_path}.total_data_bytes must equal total_units*bytes_per_unit")
        elif bpu != 1 or total_bytes > total_units:
            raise PipelineDescriptorError(f"{shard_path} has invalid stream_bytes extent/bytes_per_unit")
        shards.append(PipelineShardRule(ordinal, source_rank, target_rank, total_bytes, bpu, path_value))
    if unit == "stream_bytes" and max(shard.total_data_bytes for shard in shards) != total_units:
        raise PipelineDescriptorError(f"{path}.tile_axis.total_units must equal max shard bytes")
    layer_start = _uint(raw["layer_start"], path + ".layer_start")
    layer_end = _positive(raw["layer_end"], path + ".layer_end")
    if layer_end <= layer_start:
        raise PipelineDescriptorError(f"{path} layer range must be nonempty")
    return PipelineGroup(
        _uint(raw["group_ordinal"], path + ".group_ordinal"),
        _uint(raw["pass_or_window_ordinal"], path + ".pass_or_window_ordinal"),
        _uint(raw["source_ordinal"], path + ".source_ordinal"),
        _string(raw["source_key"], path + ".source_key"),
        layer_start, layer_end, axis, tuple(shards))


def _parse_protocol(value: Any) -> PipelineProtocol:
    path = "protocol"
    raw = _mapping(value, path, ("buffer_slots", "release_lag_tiles", "initial_tiles_without_release",
        "fill_ack_bytes", "release_ack_bytes", "hbm_charge", "target_rank_order", "source_rank_order", "tile_branching"))
    protocol = PipelineProtocol(
        _positive(raw["buffer_slots"], path + ".buffer_slots", 0xFFFFFFFF),
        _positive(raw["release_lag_tiles"], path + ".release_lag_tiles", 0xFFFFFFFF),
        _positive(raw["initial_tiles_without_release"], path + ".initial_tiles_without_release", 0xFFFFFFFF),
        _positive(raw["fill_ack_bytes"], path + ".fill_ack_bytes"),
        _positive(raw["release_ack_bytes"], path + ".release_ack_bytes"),
        raw["hbm_charge"],
        _string(raw["target_rank_order"], path + ".target_rank_order"),
        _string(raw["source_rank_order"], path + ".source_rank_order"),
        _string(raw["tile_branching"], path + ".tile_branching"))
    if type(protocol.hbm_charge) is not bool or protocol != PipelineProtocol():
        raise PipelineDescriptorError("protocol must match the frozen two-slot lag-2 contract")
    return protocol


def _parse_prediction_profile(value: Any) -> PipelinePredictionProfile:
    raw = _mapping(value, "prediction_profile", ("prefill_consume", "decode_consume"))
    prefill = raw["prefill_consume"]
    if prefill != "zero_duration_no_extra_consume":
        raise PipelineDescriptorError("prediction_profile.prefill_consume is fixed")
    decode = _mapping(raw["decode_consume"], "prediction_profile.decode_consume",
        ("kind", "profile_id", "compute_ns_total", "hbm_bytes_total", "resource_key", "share_rule"))
    compute = decode["compute_ns_total"]
    if type(compute) not in (int, float) or not math.isfinite(compute) or compute < 0:
        raise PipelineDescriptorError("decode compute_ns_total must be finite and non-negative")
    profile = PipelineDecodePredictionProfile(
        _enum(decode["kind"], "decode_consume.kind", ("profiled_consume",)),
        _string(decode["profile_id"], "decode_consume.profile_id"), float(compute),
        _uint(decode["hbm_bytes_total"], "decode_consume.hbm_bytes_total"),
        _string(decode["resource_key"], "decode_consume.resource_key"),
        _enum(decode["share_rule"], "decode_consume.share_rule", ("existing_action_work_graph_byte_share_vector",)))
    return PipelinePredictionProfile(prefill, profile)


def _parse_execution_profile(value: Any) -> PipelineExecutionProfile:
    raw = _mapping(value, "execution_profile", ("data_legs", "endpoint_hbm_rule", "prefill_consume", "decode_consume"))
    result = PipelineExecutionProfile(*(
        _string(raw[key], "execution_profile." + key)
        for key in ("data_legs", "endpoint_hbm_rule", "prefill_consume", "decode_consume")))
    expected = PipelineExecutionProfile(
        "ordinary_comm_send_recv_path",
        "one_original_COMM_READ_or_COMM_WRITE_job_per_charged_endpoint",
        "existing_body_or_marker_only", "existing_body_nodes_only")
    if result != expected:
        raise PipelineDescriptorError("execution_profile must preserve original COMM/HBM/body semantics")
    return result


def _parse_id_range(value: Any, path: str) -> PipelineLogicalIdRange:
    raw = _mapping(value, path, ("rank", "first_json_id", "count", "stride", "kind"))
    rank = _rank(raw["rank"], path + ".rank")
    first = _uint(raw["first_json_id"], path + ".first_json_id")
    count = _positive(raw["count"], path + ".count")
    stride = _positive(raw["stride"], path + ".stride")
    # Do the multiplication/addition in Python's unbounded integer domain (the
    # wire equivalent of the required uint128 intermediate), then constrain
    # the resulting JSON IDs to the uint64 domain used by NodeStore.
    last = first + stride * (count - 1)
    if last > U64_MAX:
        raise PipelineDescriptorError(f"{path} terminal ID exceeds uint64")
    return PipelineLogicalIdRange(
        rank, first, count, stride,
        _enum(raw["kind"], path + ".kind", ("virtual_endpoint_hole",)))


def _validate_id_ranges(ranges: Tuple[PipelineLogicalIdRange, ...]) -> None:
    last_rank = -1
    last_first = {}
    # Per-rank active list of (last_id, progression).  Ranges arrive sorted by
    # first_json_id within a rank (checked below before the pairwise test), so
    # a prior whose last ID is below the current first can never intersect the
    # current or any later range and is retired here.  This keeps the pairwise
    # overlap validation linear for the disjoint contiguous ranges the
    # emitters produce while preserving exactly which inputs raise.
    active_by_rank = {}
    for item in ranges:
        if item.rank < last_rank:
            raise PipelineDescriptorError("logical ID ranges must be rank ordered")
        if item.rank in last_first and item.first_json_id < last_first[item.rank]:
            raise PipelineDescriptorError("logical ID ranges must be ordered by first_json_id per rank")
        current = (item.rank, item.first_json_id, item.stride, item.count)
        last_id = item.first_json_id + item.stride * (item.count - 1)
        actives = active_by_rank.get(item.rank)
        if actives is None:
            actives = active_by_rank[item.rank] = []
        kept = []
        for prior_last, prior in actives:
            if prior_last < item.first_json_id:
                continue
            if _progressions_intersect(current, prior):
                raise PipelineDescriptorError("logical ID arithmetic progressions overlap")
            kept.append((prior_last, prior))
        kept.append((last_id, current))
        active_by_rank[item.rank] = kept
        last_first[item.rank] = item.first_json_id
        last_rank = item.rank


def _parse_tag_run(value: Any, path: str) -> PipelineTagAllocationRun:
    raw = _mapping(value, path, ("allocator_scope_id", "emission_ordinal", "first_tag", "count",
        "group_ordinal", "first_tile", "first_shard", "tile_count", "shard_count"))
    return PipelineTagAllocationRun(
        _string(raw["allocator_scope_id"], path + ".allocator_scope_id"),
        _uint(raw["emission_ordinal"], path + ".emission_ordinal"),
        _positive(raw["first_tag"], path + ".first_tag", U32_MAX),
        _positive(raw["count"], path + ".count", U32_MAX),
        _uint(raw["group_ordinal"], path + ".group_ordinal"),
        _positive(raw["first_tile"], path + ".first_tile"),
        _uint(raw["first_shard"], path + ".first_shard"),
        _positive(raw["tile_count"], path + ".tile_count"),
        _positive(raw["shard_count"], path + ".shard_count"))


_ENDPOINTS = ("data_send", "data_recv", "fill_ack_send", "fill_ack_recv",
              "release_ack_send", "release_ack_recv", "body_marker")


def _endpoint_owner_rank(shard: PipelineShardRule, endpoint: str) -> int:
    """Return the rank that owns an emitted endpoint under legacy wiring.

    Data flows source -> target. Both acknowledgement sends reverse that
    direction, so ACK sends are target-side and ACK receives source-side.
    A body marker is a target-side queue/consume event.
    """
    if endpoint not in _ENDPOINTS:
        raise PipelineDescriptorError(f"unknown endpoint kind {endpoint!r}")
    source_side = endpoint in (
        "data_send", "fill_ack_recv", "release_ack_recv")
    return shard.source_rank if source_side else shard.target_rank


def _parse_endpoint_run(value: Any, path: str) -> PipelineEndpointRun:
    raw = _mapping(value, path, ("rank", "group_ordinal", "first_tile", "tile_count",
        "shard_ordinals", "first_json_id", "tile_stride", "shard_stride",
        "initial_parent_json_id", "legacy_context", "legs"))
    shards = tuple(_uint(item, f"{path}.shard_ordinals[{i}]")
                   for i, item in enumerate(_array(raw["shard_ordinals"],
                                                    path + ".shard_ordinals", nonempty=True)))
    if tuple(sorted(set(shards))) != shards:
        raise PipelineDescriptorError(f"{path}.shard_ordinals must be strictly increasing")
    parent = raw["initial_parent_json_id"]
    if parent is not None:
        parent = _uint(parent, path + ".initial_parent_json_id")
    context_raw = _mapping(raw["legacy_context"], path + ".legacy_context",
                           ("request_id", "stage", "generation"))
    context_stage = _enum(context_raw["stage"], path + ".legacy_context.stage",
                          ("prefill", "decode", "completion"))
    context_generation = _uint(
        context_raw["generation"], path + ".legacy_context.generation")
    if context_generation != (0 if context_stage == "prefill" else 1):
        raise PipelineDescriptorError(
            f"{path}.legacy_context generation disagrees with stage")
    legacy_context = PipelineLegacyContext(
        _string(context_raw["request_id"], path + ".legacy_context.request_id"),
        context_stage, context_generation)
    legs = []
    for i, value in enumerate(_array(raw["legs"], path + ".legs", nonempty=True)):
        leg_path = f"{path}.legs[{i}]"
        leg = _mapping(value, leg_path, ("endpoint", "json_offset", "tag_emission_ordinal",
            "tag_offset", "tag_tile_stride", "tag_shard_stride"))
        endpoint = _enum(leg["endpoint"], leg_path + ".endpoint", _ENDPOINTS)
        json_offset = _uint(leg["json_offset"], leg_path + ".json_offset")
        tag_values = tuple(leg[key] for key in
                           ("tag_emission_ordinal", "tag_offset", "tag_tile_stride", "tag_shard_stride"))
        if endpoint == "body_marker":
            if any(item is not None for item in tag_values):
                raise PipelineDescriptorError(f"{leg_path} body_marker tag fields must be null")
            emission = tag_offset = tag_tile_stride = tag_shard_stride = None
        else:
            if any(item is None for item in tag_values):
                raise PipelineDescriptorError(f"{leg_path} requires all tag fields")
            emission = _uint(leg["tag_emission_ordinal"], leg_path + ".tag_emission_ordinal")
            tag_offset = _uint(leg["tag_offset"], leg_path + ".tag_offset", U32_MAX)
            tag_tile_stride = _positive(leg["tag_tile_stride"], leg_path + ".tag_tile_stride")
            tag_shard_stride = _positive(leg["tag_shard_stride"], leg_path + ".tag_shard_stride")
        legs.append(PipelineEndpointLegRun(endpoint, json_offset, emission,
                                           tag_offset, tag_tile_stride, tag_shard_stride))
    endpoints = tuple(leg.endpoint for leg in legs)
    if len(set(endpoints)) != len(endpoints):
        raise PipelineDescriptorError(f"{path}.legs endpoint kinds must be unique")
    offsets = tuple(leg.json_offset for leg in legs)
    if len(set(offsets)) != len(offsets):
        raise PipelineDescriptorError(f"{path}.legs json_offset values must be unique")
    return PipelineEndpointRun(
        _rank(raw["rank"], path + ".rank"),
        _uint(raw["group_ordinal"], path + ".group_ordinal"),
        _positive(raw["first_tile"], path + ".first_tile"),
        _positive(raw["tile_count"], path + ".tile_count"),
        shards,
        _uint(raw["first_json_id"], path + ".first_json_id"),
        _positive(raw["tile_stride"], path + ".tile_stride"),
        _positive(raw["shard_stride"], path + ".shard_stride"),
        parent, legacy_context, tuple(legs))


def _validate_endpoint_runs(endpoint_runs: Tuple[PipelineEndpointRun, ...],
                            groups: Tuple[PipelineGroup, ...],
                            tag_runs: Tuple[PipelineTagAllocationRun, ...],
                            _id_ranges: Tuple[PipelineLogicalIdRange, ...]) -> None:
    groups_by_id = {group.group_ordinal: group for group in groups}
    tag_by_emission = {}
    for tag_run in tag_runs:
        if tag_run.emission_ordinal in tag_by_emission:
            raise PipelineDescriptorError("tag emission ordinals must be globally unique")
        tag_by_emission[tag_run.emission_ordinal] = tag_run
    for index, run in enumerate(endpoint_runs):
        group = groups_by_id.get(run.group_ordinal)
        if group is None:
            raise PipelineDescriptorError("endpoint run references unknown group")
        tile_end = _checked_add(run.first_tile, run.tile_count - 1,
                                "endpoint tile range")
        if tile_end > group.tile_axis.tile_count:
            raise PipelineDescriptorError("endpoint run tile range exceeds group")
        if any(shard >= len(group.shards) for shard in run.shard_ordinals):
            raise PipelineDescriptorError("endpoint run shard is outside group")
        axis = group.tile_axis
        for shard_index in run.shard_ordinals:
            shard = group.shards[shard_index]
            active_count = (
                axis.tile_count if axis.unit == "source_tokens"
                else min(axis.tile_count,
                         _ceil_div(shard.total_data_bytes,
                                   axis.units_per_tile)))
            if (any(leg.endpoint != "body_marker" for leg in run.legs)
                    and tile_end > active_count):
                raise PipelineDescriptorError(
                    "COMM endpoint run exceeds this shard's active tile cap")
        for leg in run.legs:
            if leg.endpoint == "body_marker":
                continue
            expected_ranks = {
                _endpoint_owner_rank(group.shards[shard], leg.endpoint)
                for shard in run.shard_ordinals}
            if expected_ranks != {run.rank}:
                raise PipelineDescriptorError("endpoint run rank disagrees with shard route")
        max_shard_local = len(run.shard_ordinals) - 1
        max_leg_offset = max(leg.json_offset for leg in run.legs)
        if run.shard_stride <= max_leg_offset:
            raise PipelineDescriptorError("endpoint shard_stride overlaps leg IDs")
        max_within_tile = _checked_add(
            _checked_mul(max_shard_local, run.shard_stride, "endpoint shard ID extent"),
            max_leg_offset, "endpoint ID extent")
        if run.tile_stride <= max_within_tile:
            raise PipelineDescriptorError("endpoint tile_stride overlaps tile IDs")
        maximum_id = _checked_add(
            run.first_json_id,
            _checked_add(_checked_mul(run.tile_count - 1, run.tile_stride,
                                      "endpoint tile ID extent"),
                         max_within_tile, "endpoint run ID extent"),
            "endpoint run ID extent")
        for leg in run.legs:
            if leg.tag_emission_ordinal is None:
                continue
            tag_run = tag_by_emission.get(leg.tag_emission_ordinal)
            if tag_run is None or tag_run.group_ordinal != run.group_ordinal:
                raise PipelineDescriptorError("endpoint leg references missing tag run")
            tag_tile_end = _checked_add(tag_run.first_tile, tag_run.tile_count - 1,
                                        "tag run tile range")
            if not (tag_run.first_tile <= run.first_tile and tile_end <= tag_tile_end):
                raise PipelineDescriptorError("endpoint tag run does not cover tile range")
            if any(shard < tag_run.first_shard or
                   shard >= tag_run.first_shard + tag_run.shard_count
                   for shard in run.shard_ordinals):
                raise PipelineDescriptorError("endpoint tag run does not cover shard range")
            max_tile_local = run.tile_count - 1
            max_shard_local = len(run.shard_ordinals) - 1
            max_tag_offset = _checked_add(
                leg.tag_offset,
                _checked_add(_checked_mul(max_tile_local, leg.tag_tile_stride,
                                          "endpoint tile tag extent"),
                             _checked_mul(max_shard_local, leg.tag_shard_stride,
                                          "endpoint shard tag extent"),
                             "endpoint tag extent"),
                "endpoint tag extent")
            if max_tag_offset >= tag_run.count:
                raise PipelineDescriptorError("endpoint tag mapping exceeds allocation run")
            _checked_add(tag_run.first_tag, max_tag_offset, "endpoint tag value")
        _ = maximum_id  # The progression checker below checks every affine leg.
        # Preserve input order as the legacy emission order. Remote-read tails
        # are emitted before their main-chain tile (tile 2 precedes tile 1),
        # so logical tile order is not a valid run sort key.


_DEPENDENCY_KINDS = ("ordinary_node_terminal", "virtual_endpoint_terminal")


def _parse_affine_dependency_run(value: Any, path: str, *, endpoint: bool):
    keys = ["rank", "group_ordinal", "first_tile", "tile_count",
            "shard_ordinals"]
    if endpoint:
        keys.append("endpoint")
    keys.extend(("dependency_kind", "first_dependency_json_id",
                 "dependency_tile_stride", "dependency_shard_stride"))
    raw = _mapping(value, path, tuple(keys))
    shards = tuple(
        _uint(item, f"{path}.shard_ordinals[{i}]")
        for i, item in enumerate(_array(
            raw["shard_ordinals"], path + ".shard_ordinals", nonempty=True)))
    if tuple(sorted(set(shards))) != shards:
        raise PipelineDescriptorError(
            f"{path}.shard_ordinals must be strictly increasing")
    endpoint_name = None
    if endpoint:
        endpoint_name = _enum(raw["endpoint"], path + ".endpoint", _ENDPOINTS)
    result = (
        _rank(raw["rank"], path + ".rank"),
        _uint(raw["group_ordinal"], path + ".group_ordinal"),
        _positive(raw["first_tile"], path + ".first_tile"),
        _positive(raw["tile_count"], path + ".tile_count"),
        shards,
        _enum(raw["dependency_kind"], path + ".dependency_kind",
              _DEPENDENCY_KINDS),
        _uint(raw["first_dependency_json_id"],
              path + ".first_dependency_json_id"),
        _uint(raw["dependency_tile_stride"],
              path + ".dependency_tile_stride"),
        _uint(raw["dependency_shard_stride"],
              path + ".dependency_shard_stride"),
    )
    return result[:5] + (endpoint_name,) + result[5:] if endpoint else result


def _parse_endpoint_dependency_run(
        value: Any, path: str) -> PipelineEndpointDependencyRun:
    return PipelineEndpointDependencyRun(
        *_parse_affine_dependency_run(value, path, endpoint=True))


def _parse_consume_completion_run(
        value: Any, path: str) -> PipelineConsumeCompletionRun:
    return PipelineConsumeCompletionRun(
        *_parse_affine_dependency_run(value, path, endpoint=False))


def _validate_dependency_run_domain(*, rank, group_ordinal, first_tile,
                                    tile_count, shard_ordinals, groups,
                                    path):
    group = groups.get(group_ordinal)
    if group is None:
        raise PipelineDescriptorError(f"{path} references an unknown group")
    if first_tile + tile_count - 1 > group.tile_axis.tile_count:
        raise PipelineDescriptorError(f"{path} exceeds its group tile axis")
    if any(shard >= len(group.shards) for shard in shard_ordinals):
        raise PipelineDescriptorError(f"{path} references an unknown shard")
    return group


def _validate_affine_dependency_extent(first_id, tile_stride, shard_stride,
                                      tile_count, shard_count, path):
    maximum = _checked_add(
        first_id,
        _checked_add(
            _checked_mul(tile_count - 1, tile_stride,
                         path + " tile dependency extent"),
            _checked_mul(shard_count - 1, shard_stride,
                         path + " shard dependency extent"),
            path + " dependency extent"),
        path + " dependency extent")
    return maximum


def _validate_endpoint_dependency_runs(dependency_runs, endpoint_runs,
                                       groups):
    groups_by_id = {group.group_ordinal: group for group in groups}
    runs_by_position = {}
    for run in endpoint_runs:
        for leg in run.legs:
            for shard in run.shard_ordinals:
                key = (run.rank, run.group_ordinal, leg.endpoint, shard)
                runs_by_position.setdefault(key, []).append(
                    (run.first_tile, run.first_tile + run.tile_count - 1))
    seen = set()
    for index, run in enumerate(dependency_runs):
        path = f"endpoint_dependency_runs[{index}]"
        group = _validate_dependency_run_domain(
            rank=run.rank, group_ordinal=run.group_ordinal,
            first_tile=run.first_tile, tile_count=run.tile_count,
            shard_ordinals=run.shard_ordinals, groups=groups_by_id,
            path=path)
        for shard_index in run.shard_ordinals:
            shard = group.shards[shard_index]
            axis = group.tile_axis
            active_count = (
                axis.tile_count if axis.unit == "source_tokens"
                else min(axis.tile_count,
                         _ceil_div(shard.total_data_bytes,
                                   axis.units_per_tile)))
            if run.endpoint != "body_marker" and (
                    run.first_tile + run.tile_count - 1 > active_count):
                raise PipelineDescriptorError(
                    f"{path} exceeds this shard's active tile cap")
            expected = _endpoint_owner_rank(shard, run.endpoint)
            if expected != run.rank:
                raise PipelineDescriptorError(
                    f"{path}.rank disagrees with endpoint route")
            pieces = runs_by_position.get(
                (run.rank, run.group_ordinal, run.endpoint, shard_index), ())
            if not any(first <= run.first_tile and
                       run.first_tile + run.tile_count - 1 <= last
                       for first, last in pieces):
                raise PipelineDescriptorError(
                    f"{path} has no matching emitted endpoint run")
        _validate_affine_dependency_extent(
            run.first_dependency_json_id, run.dependency_tile_stride,
            run.dependency_shard_stride, run.tile_count,
            len(run.shard_ordinals), path)
        key = (run.rank, run.group_ordinal, run.first_tile, run.tile_count,
               run.shard_ordinals, run.endpoint, run.dependency_kind,
               run.first_dependency_json_id, run.dependency_tile_stride,
               run.dependency_shard_stride)
        if key in seen:
            raise PipelineDescriptorError(f"{path} duplicates an earlier run")
        seen.add(key)


def _validate_consume_completion_runs(completion_runs, groups, identity_rank):
    groups_by_id = {group.group_ordinal: group for group in groups}
    covered = {}
    for index, run in enumerate(completion_runs):
        path = f"consume_completion_runs[{index}]"
        group = _validate_dependency_run_domain(
            rank=run.rank, group_ordinal=run.group_ordinal,
            first_tile=run.first_tile, tile_count=run.tile_count,
            shard_ordinals=run.shard_ordinals, groups=groups_by_id,
            path=path)
        if run.rank != identity_rank:
            raise PipelineDescriptorError(
                f"{path}.rank must equal the descriptor buffer owner rank")
        for shard_index in run.shard_ordinals:
            if group.shards[shard_index].target_rank != run.rank:
                raise PipelineDescriptorError(
                    f"{path}.rank must be the target-side owner rank")
            key = (run.group_ordinal, shard_index)
            covered.setdefault(key, []).append(
                (run.first_tile, run.first_tile + run.tile_count - 1))
        _validate_affine_dependency_extent(
            run.first_dependency_json_id, run.dependency_tile_stride,
            run.dependency_shard_stride, run.tile_count,
            len(run.shard_ordinals), path)
    # Every target-side tile that has bytes needs the exact body-tail or
    # queue-accept terminal that releases its staging credit. Shared stream
    # axes can outlive a short shard tail; those trailing zero-byte slots have
    # no physical endpoint and must not be padded into completion coverage.
    # Validate intervals only; the range length may be billions of tiles.
    for group in groups:
        axis = group.tile_axis
        for shard_index, shard in enumerate(group.shards):
            if shard.target_rank != identity_rank:
                continue
            if axis.unit == "source_tokens":
                active_tiles = axis.tile_count if shard.bytes_per_unit else 0
            else:
                active_tiles = min(
                    axis.tile_count,
                    _ceil_div(shard.total_data_bytes, axis.units_per_tile))
            intervals = sorted(covered.get((group.group_ordinal, shard_index), ()))
            next_tile = 1
            for first, last in intervals:
                if first != next_tile:
                    raise PipelineDescriptorError(
                        "consume completion runs have a gap or overlap for "
                        f"group {group.group_ordinal}, shard {shard_index}")
                next_tile = last + 1
            if next_tile != active_tiles + 1:
                raise PipelineDescriptorError(
                    "consume completion runs do not cover every active target tile "
                    f"for group {group.group_ordinal}, shard {shard_index}")


def _progressions_intersect(left, right) -> bool:
    left_rank, left_first, left_stride, left_count = left
    right_rank, right_first, right_stride, right_count = right
    if left_rank != right_rank:
        return False
    low = max(left_first, right_first)
    high = min(left_first + left_stride * (left_count - 1),
               right_first + right_stride * (right_count - 1))
    if low > high:
        return False
    common = math.gcd(left_stride, right_stride)
    delta = right_first - left_first
    if delta % common:
        return False
    modulus = right_stride // common
    step = 0 if modulus == 1 else (
        ((delta // common) % modulus)
        * pow((left_stride // common) % modulus, -1, modulus)) % modulus
    first = left_first + left_stride * step
    period = (left_stride // common) * right_stride
    if first < low:
        first += _ceil_div(low - first, period) * period
    return first <= high


def _progression_intersection_count(left, right) -> int:
    """Count common IDs of two finite arithmetic progressions exactly."""
    left_rank, left_first, left_stride, left_count = left
    right_rank, right_first, right_stride, right_count = right
    if left_rank != right_rank or not left_count or not right_count:
        return 0
    low = max(left_first, right_first)
    high = min(left_first + left_stride * (left_count - 1),
               right_first + right_stride * (right_count - 1))
    if low > high:
        return 0
    common = math.gcd(left_stride, right_stride)
    delta = right_first - left_first
    if delta % common:
        return 0
    modulus = right_stride // common
    step = 0 if modulus == 1 else (
        ((delta // common) % modulus)
        * pow((left_stride // common) % modulus, -1, modulus)) % modulus
    first = left_first + left_stride * step
    period = (left_stride // common) * right_stride
    if first < low:
        first += _ceil_div(low - first, period) * period
    if first > high:
        return 0
    return (high - first) // period + 1


class _EndpointIdIndex:
    """Per-rank hole/anchor index for endpoint-ID coverage validation.

    Holes are stored per rank sorted by ``first_json_id`` together with a
    prefix-max of their last IDs.  A query only walks the suffix of holes
    whose first ID is within the progression range and stops as soon as the
    prefix-max proves no earlier hole can still overlap.  Anchors are stored
    per rank as a sorted ID list, so a query reads only the anchors inside
    the progression range.  Both restrictions are exact: rows outside the
    range contribute nothing to the covered count in the original scan.
    """

    __slots__ = ("_holes_by_rank", "_hole_firsts_by_rank", "_hole_prefix_max",
                 "_anchors_by_rank")

    def __init__(self, holes, anchors):
        by_rank = {}
        for hole in holes:
            by_rank.setdefault(hole.rank, []).append(
                (hole.first_json_id, hole.first_json_id
                 + hole.stride * (hole.count - 1), hole))
        self._holes_by_rank = {}
        self._hole_firsts_by_rank = {}
        self._hole_prefix_max = {}
        for rank, items in by_rank.items():
            items.sort(key=lambda entry: entry[0])
            self._holes_by_rank[rank] = items
            self._hole_firsts_by_rank[rank] = [entry[0] for entry in items]
            prefix_max = []
            running = None
            for _first, last, _hole in items:
                running = last if running is None else max(running, last)
                prefix_max.append(running)
            self._hole_prefix_max[rank] = prefix_max
        anchors_by_rank = {}
        for anchor in anchors:
            if (anchor.binding != "physical_endpoint_slot"
                    or anchor.node_id is None):
                continue
            anchors_by_rank.setdefault(anchor.rank, []).append(anchor.node_id)
        self._anchors_by_rank = {
            rank: sorted(ids) for rank, ids in anchors_by_rank.items()}

    @staticmethod
    def _bisect_right(values, bound):
        lo, hi = 0, len(values)
        while lo < hi:
            mid = (lo + hi) // 2
            if values[mid] <= bound:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def covered_count(self, rank, first, stride, count):
        total = 0
        last = first + stride * (count - 1)
        items = self._holes_by_rank.get(rank)
        if items:
            prefix_max = self._hole_prefix_max[rank]
            # holes with first <= last form a prefix of the sorted list
            stop = self._bisect_right(self._hole_firsts_by_rank[rank], last)
            progression = (rank, first, stride, count)
            for index in range(stop - 1, -1, -1):
                if prefix_max[index] < first:
                    break
                hole_first, hole_last, hole = items[index]
                if hole_last < first or hole_first > last:
                    continue
                total += _progression_intersection_count(
                    progression,
                    (hole.rank, hole.first_json_id, hole.stride, hole.count))
        anchor_ids = self._anchors_by_rank.get(rank)
        if anchor_ids:
            lo = self._bisect_right(anchor_ids, first - 1)
            hi = self._bisect_right(anchor_ids, last)
            for index in range(lo, hi):
                delta = anchor_ids[index] - first
                if delta % stride == 0:
                    total += 1
        return total

    def anchor_hits_hole(self, rank, node_id):
        """Exact singleton-vs-hole membership (no modular-inverse work)."""
        items = self._holes_by_rank.get(rank)
        if not items:
            return False
        prefix_max = self._hole_prefix_max[rank]
        stop = self._bisect_right(self._hole_firsts_by_rank[rank], node_id)
        for index in range(stop - 1, -1, -1):
            hole_first, hole_last, hole = items[index]
            if hole_last < node_id:
                if prefix_max[index] < node_id:
                    break
                continue
            if (node_id >= hole_first
                    and (node_id - hole_first) % hole.stride == 0):
                return True
        return False


def _progression_covered(progression, holes, anchors) -> bool:
    rank, first, stride, count = progression
    covered_count = _EndpointIdIndex(holes, anchors).covered_count(
        rank, first, stride, count)
    # ID holes are pairwise disjoint, and the anchor validator below rejects
    # physical IDs that alias holes or each other, so the sum is a union size.
    return covered_count == count


def _validate_endpoint_id_targets(endpoint_runs, id_ranges, anchors) -> None:
    index = _EndpointIdIndex(id_ranges, anchors)
    for anchor in anchors:
        if anchor.binding != "physical_endpoint_slot" or anchor.node_id is None:
            continue
        if index.anchor_hits_hole(anchor.rank, anchor.node_id):
            raise PipelineDescriptorError("physical endpoint anchor aliases a virtual ID hole")
    progressions = []
    for run_index, run in enumerate(endpoint_runs):
        for shard_index, _shard in enumerate(run.shard_ordinals):
            for leg_index, leg in enumerate(run.legs):
                first = _checked_add(
                    run.first_json_id,
                    _checked_add(_checked_mul(shard_index, run.shard_stride,
                                              "endpoint shard progression"),
                                 leg.json_offset,
                                 "endpoint leg progression"),
                    "endpoint progression")
                progression = (run.rank, first, run.tile_stride,
                               run.tile_count)
                if index.covered_count(
                        run.rank, first, run.tile_stride,
                        run.tile_count) != run.tile_count:
                    raise PipelineDescriptorError(
                        f"endpoint_runs[{run_index}] shard {shard_index} leg "
                        f"{leg_index} maps an ID neither to a virtual hole nor "
                        "a physical anchor")
                progressions.append((progression, run_index, shard_index, leg_index))
    # Pairwise overlap check with a range sweep: two arithmetic progressions
    # can only share an ID when their ID ranges overlap, so entries whose
    # range ends below the current start are retired.  The emitters produce
    # disjoint contiguous extents, which keeps the active set at O(1); the
    # check raises for exactly the same inputs as the full pairwise scan.
    sweep = []
    for position, (progression, *_label) in enumerate(progressions):
        rank, first, stride, count = progression
        sweep.append((first, first + stride * (count - 1), position,
                      progression))
    sweep.sort(key=lambda entry: entry[0])
    actives = []
    for first, last, position, progression in sweep:
        kept = []
        for active_last, active_position, active in actives:
            if active_last < first:
                continue
            if _progressions_intersect(active, progression):
                raise PipelineDescriptorError("affine endpoint IDs overlap")
            kept.append((active_last, active_position, active))
        kept.append((last, position, progression))
        actives = kept


_SHARE_RULE = "quotient_remainder_earliest_global_indices"


def _string_array(value: Any, path: str) -> Tuple[str, ...]:
    items = _array(value, path)
    return tuple(_string(item, f"{path}[{i}]") for i, item in enumerate(items))


def _completion_order(raw: Mapping[str, Any], path: str, tile_count: int
                     ) -> Tuple[Optional[int], int]:
    first = raw["completion_order_first"]
    if first is not None:
        first = _int64(first, path + ".completion_order_first")
    stride = _int64(raw["completion_order_stride"], path + ".completion_order_stride")
    if first is None:
        if stride != 0:
            raise PipelineDescriptorError(
                f"{path} null completion_order_first requires zero stride")
    else:
        last = first + stride * (tile_count - 1)
        if not I64_MIN <= last <= I64_MAX:
            raise PipelineDescriptorError(
                f"{path} completion-order progression overflows int64")
    return first, stride


def _parse_projection_share(value: Any, path: str) -> Optional[PipelineProjectionShare]:
    if value is None:
        return None
    raw = _mapping(value, path, ("rule", "total_units", "total_tiles",
                                 "share_index_first", "tile_bytes"))
    return PipelineProjectionShare(
        _enum(raw["rule"], path + ".rule", (_SHARE_RULE,)),
        _uint(raw["total_units"], path + ".total_units"),
        _positive(raw["total_tiles"], path + ".total_tiles"),
        _uint(raw["share_index_first"], path + ".share_index_first"),
        _positive(raw["tile_bytes"], path + ".tile_bytes"))


def _parse_projection_fill(value: Any, path: str, tile_count: int,
                           protocol: PipelineProtocol) -> PipelineProjectionFill:
    raw = _mapping(value, path, (
        "leg_id_prefix", "first_index", "fill_order_first", "fill_order_stride",
        "completion_order_first", "completion_order_stride", "route",
        "bytes_per_tile", "latency_ns", "start_deps", "gate_id",
        "gate_capacity"))
    first, completion_stride = _completion_order(raw, path, tile_count)
    return PipelineProjectionFill(
        _string(raw["leg_id_prefix"], path + ".leg_id_prefix"),
        _uint(raw["first_index"], path + ".first_index"),
        _uint(raw["fill_order_first"], path + ".fill_order_first"),
        _positive(raw["fill_order_stride"], path + ".fill_order_stride"),
        first, completion_stride,
        _string_array(raw["route"], path + ".route"),
        _positive(raw["bytes_per_tile"], path + ".bytes_per_tile"),
        _uint(raw["latency_ns"], path + ".latency_ns"),
        _string_array(raw["start_deps"], path + ".start_deps"),
        _string(raw["gate_id"], path + ".gate_id"),
        _positive(raw["gate_capacity"], path + ".gate_capacity", U32_MAX))


def _parse_projection_consume(value: Any, path: str, tile_count: int
                              ) -> PipelineProjectionConsume:
    raw = _mapping(value, path, (
        "leg_id_prefix", "first_index", "consume_order_first",
        "consume_order_stride", "completion_order_first",
        "completion_order_stride", "kind", "route", "group",
        "latency_first_ns", "latency_rest_ns", "consume_deps",
        "consume_start_deps", "gate_id", "compute_share", "hbm_share"))
    first, completion_stride = _completion_order(raw, path, tile_count)
    kind = _enum(raw["kind"], path + ".kind", ("fixed", "compute"))
    group = raw["group"]
    if group is not None:
        group = _string(group, path + ".group")
    compute_share = _parse_projection_share(raw["compute_share"], path + ".compute_share")
    hbm_share = _parse_projection_share(raw["hbm_share"], path + ".hbm_share")
    route = _string_array(raw["route"], path + ".route")
    latency_first = _uint(raw["latency_first_ns"], path + ".latency_first_ns")
    latency_rest = _uint(raw["latency_rest_ns"], path + ".latency_rest_ns")
    if kind == "fixed":
        if (route or group is not None or latency_first or latency_rest
                or compute_share is not None or hbm_share is not None):
            raise PipelineDescriptorError(
                f"{path} fixed consume must have empty route, null group/shares, and zero latency")
    else:
        if not route or group != "decode":
            raise PipelineDescriptorError(
                f"{path} compute consume requires a route and decode group")
        if compute_share is None or hbm_share is None:
            raise PipelineDescriptorError(
                f"{path} compute consume requires separate compute_share and hbm_share")
        for name, share in (("compute_share", compute_share), ("hbm_share", hbm_share)):
            if (share.total_tiles != compute_share.total_tiles
                    or share.share_index_first + tile_count > share.total_tiles):
                raise PipelineDescriptorError(
                    f"{path}.{name} tile range is inconsistent with the chain")
        if (compute_share.tile_bytes != hbm_share.tile_bytes
                or compute_share.share_index_first != hbm_share.share_index_first):
            raise PipelineDescriptorError(
                f"{path} compute and HBM shares must use the same global byte positions")
    return PipelineProjectionConsume(
        _string(raw["leg_id_prefix"], path + ".leg_id_prefix"),
        _uint(raw["first_index"], path + ".first_index"),
        _uint(raw["consume_order_first"], path + ".consume_order_first"),
        _positive(raw["consume_order_stride"], path + ".consume_order_stride"),
        first, completion_stride, kind, route, group, latency_first,
        latency_rest,
        _string_array(raw["consume_deps"], path + ".consume_deps"),
        _string_array(raw["consume_start_deps"], path + ".consume_start_deps"),
        _string(raw["gate_id"], path + ".gate_id"), compute_share, hbm_share)


def _progression_ranges_disjoint(entries, label: str) -> None:
    for index, left in enumerate(entries):
        for right in entries[index + 1:]:
            if _progressions_intersect(left, right):
                raise PipelineDescriptorError(f"projection {label} ranges overlap")


def _validate_projection(projection: PipelineProjection,
                         groups: Tuple[PipelineGroup, ...],
                         protocol: PipelineProtocol,
                         owner_rank: int) -> None:
    groups_by_id = {group.group_ordinal: group for group in groups}
    seen_group_rank = set()
    gate_capacity = {}
    fill_orders = []
    consume_orders = []
    leg_ranges = {}
    per_rank = {}
    for index, chain in enumerate(projection.chains):
        path = f"projection.chains[{index}]"
        group = groups_by_id.get(chain.group_ordinal)
        if group is None:
            raise PipelineDescriptorError(f"{path} references unknown group")
        if chain.tile_count != group.tile_axis.tile_count:
            raise PipelineDescriptorError(f"{path} tile_count must match its group")
        key = (chain.group_ordinal, chain.rank)
        if key in seen_group_rank:
            raise PipelineDescriptorError(f"{path} duplicates a group/rank chain")
        seen_group_rank.add(key)
        routed_targets = {shard.target_rank for shard in group.shards
                          if shard.total_data_bytes > 0}
        if chain.rank not in routed_targets:
            raise PipelineDescriptorError(f"{path} rank has no charged target shard")
        if chain.rank != owner_rank:
            raise PipelineDescriptorError(
                f"{path} rank must equal descriptor buffer-domain owner rank")
        fill = chain.fill
        consume = chain.consume
        if fill.first_index != consume.first_index:
            raise PipelineDescriptorError(
                f"{path} fill and consume index ranges must align")
        if not fill.route:
            raise PipelineDescriptorError(f"{path}.fill route must not be empty")
        if fill.gate_capacity != protocol.buffer_slots:
            raise PipelineDescriptorError(
                f"{path}.fill gate capacity disagrees with the double-buffer protocol")
        if fill.gate_id != consume.gate_id:
            raise PipelineDescriptorError(f"{path} fill/consume gate IDs must match")
        prior_capacity = gate_capacity.setdefault(fill.gate_id, fill.gate_capacity)
        if prior_capacity != fill.gate_capacity:
            raise PipelineDescriptorError("a projection gate ID has conflicting capacities")
        fill_end = _checked_add(fill.first_index, chain.tile_count,
                                "projection fill ID index range")
        consume_end = _checked_add(consume.first_index, chain.tile_count,
                                   "projection consume ID index range")
        leg_ranges.setdefault(fill.leg_id_prefix, []).append(
            (fill.first_index, fill_end, path + ".fill"))
        leg_ranges.setdefault(consume.leg_id_prefix, []).append(
            (consume.first_index, consume_end, path + ".consume"))
        fill_orders.append((0, fill.fill_order_first, fill.fill_order_stride,
                            chain.tile_count))
        consume_orders.append((0, consume.consume_order_first,
                               consume.consume_order_stride, chain.tile_count))
        per_rank.setdefault(chain.rank, []).append(
            (chain.group_ordinal, fill.first_index, chain.tile_count))
    for prefix, ranges in leg_ranges.items():
        ordered = sorted(ranges)
        for previous, current in zip(ordered, ordered[1:]):
            if current[0] < previous[1]:
                raise PipelineDescriptorError(
                    f"projection leg ID ranges overlap for prefix {prefix!r}")
    _progression_ranges_disjoint(
        fill_orders + consume_orders, "registration-order")
    for rank, segments in per_rank.items():
        segments.sort()
        if segments[0][1] != 0:
            raise PipelineDescriptorError(
                f"projection rank {rank} first chain must start at index zero")
        previous_end = None
        for _group, first, count in segments:
            if previous_end is not None and first != previous_end:
                raise PipelineDescriptorError(
                    f"projection rank {rank} group chains are not contiguous")
            previous_end = first + count


def _parse_projection(value: Any, groups: Tuple[PipelineGroup, ...],
                      protocol: PipelineProtocol,
                      owner_rank: int) -> PipelineProjection:
    raw = _mapping(value, "projection", ("version", "chains"))
    version = _uint(raw["version"], "projection.version", 0xFFFFFFFF)
    if version != 1:
        raise PipelineDescriptorError(f"unsupported projection version {version}")
    chains = []
    for index, value in enumerate(_array(raw["chains"], "projection.chains")):
        path = f"projection.chains[{index}]"
        chain_raw = _mapping(value, path, (
            "group_ordinal", "rank", "tile_count", "fill", "consume"))
        tile_count = _positive(chain_raw["tile_count"], path + ".tile_count")
        chains.append(PipelineProjectionChain(
            _uint(chain_raw["group_ordinal"], path + ".group_ordinal"),
            _rank(chain_raw["rank"], path + ".rank"), tile_count,
            _parse_projection_fill(chain_raw["fill"], path + ".fill",
                                   tile_count, protocol),
            _parse_projection_consume(chain_raw["consume"], path + ".consume",
                                      tile_count)))
    result = PipelineProjection(version, tuple(chains))
    _validate_projection(result, groups, protocol, owner_rank)
    return result


def _validate_tag_runs(runs: Tuple[PipelineTagAllocationRun, ...], groups: Tuple[PipelineGroup, ...]) -> None:
    by_id = {group.group_ordinal: group for group in groups}
    last = {}
    for run in runs:
        group = by_id.get(run.group_ordinal)
        if group is None:
            raise PipelineDescriptorError("tag run references unknown group")
        if run.first_tile + run.tile_count - 1 > group.tile_axis.tile_count:
            raise PipelineDescriptorError("tag run tile range exceeds group")
        if run.first_shard + run.shard_count > len(group.shards):
            raise PipelineDescriptorError("tag run shard range exceeds group")
        tag_end = run.first_tag + run.count
        if tag_end > U32_MAX + 1:
            raise PipelineDescriptorError("tag run exceeds uint32 tag space")
        previous = last.get(run.allocator_scope_id)
        if previous and (run.emission_ordinal <= previous[0] or run.first_tag < previous[1]):
            raise PipelineDescriptorError("tag runs must preserve allocator emission/tag order")
        last[run.allocator_scope_id] = (run.emission_ordinal, tag_end)


def _parse_anchor(value: Any, path: str) -> PipelineAnchor:
    raw = _mapping(value, path, ("role", "rank", "binding", "node_id", "selector"))
    role = _enum(raw["role"], path + ".role", ("start", "frontier", "service", "settle"))
    rank = _rank(raw["rank"], path + ".rank")
    binding = _enum(raw["binding"], path + ".binding", ("physical_endpoint_slot", "descriptor_cursor_selector"))
    node_id = None if raw["node_id"] is None else _uint(raw["node_id"], path + ".node_id")
    selector_raw = _mapping(raw["selector"], path + ".selector", ("kind",), (
        "group_ordinal", "tile_ordinal", "shard_ordinal", "endpoint", "existing_gate",
        "tag_allocation_emission_ordinal", "tag_offset"))
    kind = _enum(selector_raw["kind"], path + ".selector.kind", (
        "stage_start", "tile_endpoint_leg", "tile_endpoint_terminal",
        "internal_marker_done", "existing_service_gate",
        "all_obligations_settled"))
    fields: dict[str, Any] = {"kind": kind}
    for key in ("group_ordinal", "tile_ordinal", "shard_ordinal", "tag_allocation_emission_ordinal"):
        if key in selector_raw:
            fields[key] = _uint(selector_raw[key], path + ".selector." + key)
    if "tag_offset" in selector_raw:
        fields["tag_offset"] = _uint(selector_raw["tag_offset"], path + ".selector.tag_offset", U32_MAX)
    if "endpoint" in selector_raw:
        fields["endpoint"] = _enum(selector_raw["endpoint"], path + ".selector.endpoint", (
            "data_send", "data_recv", "fill_ack_send", "fill_ack_recv", "release_ack_send", "release_ack_recv", "body_marker"))
    if "existing_gate" in selector_raw:
        fields["existing_gate"] = _string(selector_raw["existing_gate"], path + ".selector.existing_gate")
    selector = PipelineAnchorSelector(**fields)
    return PipelineAnchor(role, rank, binding, node_id, selector)


def _validate_anchors(anchors: Tuple[PipelineAnchor, ...], groups: Tuple[PipelineGroup, ...],
                      runs: Tuple[PipelineTagAllocationRun, ...],
                      endpoint_runs: Tuple[PipelineEndpointRun, ...]) -> None:
    groups_by_id = {group.group_ordinal: group for group in groups}
    runs_by_emission = {run.emission_ordinal: run for run in runs}
    role_counts = {}
    physical_slots = set()
    selectors = set()
    physical_ids = set()
    for anchor in anchors:
        role_counts[anchor.role] = role_counts.get(anchor.role, 0) + 1
        sel = anchor.selector
        selector_key = (anchor.rank, sel)
        if selector_key in selectors:
            raise PipelineDescriptorError("anchor selectors must be unique per rank")
        selectors.add(selector_key)
        endpoint = sel.kind in ("tile_endpoint_leg", "tile_endpoint_terminal",
                                "internal_marker_done")
        if anchor.binding == "physical_endpoint_slot":
            if anchor.role not in ("start", "frontier"):
                raise PipelineDescriptorError("only start/frontier anchors may bind physical endpoint slots")
            if anchor.node_id is None:
                raise PipelineDescriptorError("physical endpoint anchor requires node_id")
            slot_key = (anchor.rank, anchor.node_id)
            if slot_key in physical_slots:
                raise PipelineDescriptorError("physical endpoint slots must be unique")
            physical_slots.add(slot_key)
            physical_ids.add(slot_key)
            if endpoint:
                marker_selector = sel.kind == "internal_marker_done"
                required = (sel.group_ordinal, sel.tile_ordinal,
                            sel.shard_ordinal, sel.endpoint)
                if not marker_selector:
                    required += (sel.tag_allocation_emission_ordinal,
                                 sel.tag_offset)
                if any(value is None for value in required) or sel.tile_ordinal == 0:
                    raise PipelineDescriptorError("physical endpoint anchor requires complete one-based selector")
                if marker_selector and (
                        sel.endpoint != "body_marker"
                        or anchor.role != "frontier"
                        or sel.tag_allocation_emission_ordinal is not None
                        or sel.tag_offset is not None):
                    raise PipelineDescriptorError(
                        "internal_marker_done selector must be a frontier body_marker without tag fields")
                if not marker_selector and sel.endpoint == "body_marker":
                    raise PipelineDescriptorError(
                        "body_marker anchors must use internal_marker_done")
                group = groups_by_id.get(sel.group_ordinal)
                if group is None or sel.shard_ordinal >= len(group.shards) or sel.tile_ordinal > group.tile_axis.tile_count:
                    raise PipelineDescriptorError("physical endpoint selector is outside its group")
                if marker_selector:
                    tag_offset = None
                else:
                    run = runs_by_emission.get(sel.tag_allocation_emission_ordinal)
                    if run is None or run.group_ordinal != sel.group_ordinal:
                        raise PipelineDescriptorError("physical endpoint selector references missing tag run")
                    tag_offset = sel.tag_offset
                    if tag_offset >= run.count or not (
                        run.first_tile <= sel.tile_ordinal < run.first_tile + run.tile_count
                        and run.first_shard <= sel.shard_ordinal < run.first_shard + run.shard_count):
                        raise PipelineDescriptorError("physical endpoint selector is outside tag run")
                matches = []
                for endpoint_run_ordinal, endpoint_run in enumerate(endpoint_runs):
                    if (endpoint_run.group_ordinal != sel.group_ordinal or
                            not endpoint_run.first_tile <= sel.tile_ordinal <
                            endpoint_run.first_tile + endpoint_run.tile_count or
                            sel.shard_ordinal not in endpoint_run.shard_ordinals):
                        continue
                    for leg_ordinal, leg in enumerate(endpoint_run.legs):
                        if (leg.endpoint == sel.endpoint and
                                (marker_selector or leg.tag_emission_ordinal == sel.tag_allocation_emission_ordinal)):
                            matches.append((endpoint_run_ordinal, leg_ordinal, leg))
                if len(matches) != 1:
                    raise PipelineDescriptorError("physical endpoint selector must identify one endpoint run leg")
                endpoint_run_ordinal, leg_ordinal, leg = matches[0]
                endpoint_run = endpoint_runs[endpoint_run_ordinal]
                if endpoint_run.rank != anchor.rank:
                    raise PipelineDescriptorError("physical endpoint anchor rank disagrees with endpoint run")
                local_tile = sel.tile_ordinal - endpoint_run.first_tile
                local_shard = endpoint_run.shard_ordinals.index(sel.shard_ordinal)
                expected_node_id = _checked_add(
                    endpoint_run.first_json_id,
                    _checked_add(_checked_mul(local_tile, endpoint_run.tile_stride,
                                              "anchor tile ID offset"),
                                 _checked_mul(local_shard, endpoint_run.shard_stride,
                                              "anchor shard ID offset"),
                                 "anchor ID offset"),
                    "anchor ID")
                expected_node_id = _checked_add(expected_node_id, leg.json_offset,
                                                "anchor leg ID")
                if expected_node_id != anchor.node_id:
                    raise PipelineDescriptorError("physical endpoint node_id disagrees with endpoint ID mapping")
                if marker_selector:
                    if leg.tag_emission_ordinal is not None:
                        raise PipelineDescriptorError(
                            "internal marker endpoint must not allocate a tag")
                else:
                    expected_tag_offset = _checked_add(
                        leg.tag_offset,
                        _checked_add(_checked_mul(local_tile, leg.tag_tile_stride,
                                                  "anchor tile tag offset"),
                                     _checked_mul(local_shard, leg.tag_shard_stride,
                                                  "anchor shard tag offset"),
                                     "anchor tag offset"),
                        "anchor tag offset")
                    if expected_tag_offset != sel.tag_offset:
                        raise PipelineDescriptorError("physical endpoint tag selector disagrees with endpoint mapping")
            else:
                raise PipelineDescriptorError("physical anchor must bind a concrete endpoint selector")
        else:
            if anchor.node_id is not None:
                raise PipelineDescriptorError("cursor selector must not carry node_id")
            if anchor.role == "service" and sel.kind != "existing_service_gate":
                raise PipelineDescriptorError("cursor service anchor must select existing_service_gate")
            if anchor.role == "settle" and sel.kind != "all_obligations_settled":
                raise PipelineDescriptorError("cursor settle anchor must select all_obligations_settled")
            if anchor.role in ("service", "settle") and anchor.binding != "descriptor_cursor_selector":
                raise PipelineDescriptorError("service/settle anchors must be cursor-only")
    for singleton_role in ("service", "settle"):
        if role_counts.get(singleton_role, 0) != 1:
            raise PipelineDescriptorError(f"descriptor needs exactly one {singleton_role} anchor")


def _group_wire(group: PipelineGroup) -> dict:
    return {
        "group_ordinal": group.group_ordinal,
        "pass_or_window_ordinal": group.pass_or_window_ordinal,
        "source_ordinal": group.source_ordinal,
        "source_key": group.source_key,
        "layer_start": group.layer_start,
        "layer_end": group.layer_end,
        "tile_axis": {
            "unit": group.tile_axis.unit,
            "total_units": group.tile_axis.total_units,
            "units_per_tile": group.tile_axis.units_per_tile,
            "tile_count": group.tile_axis.tile_count,
            "tail_units": group.tile_axis.tail_units,
            "global_tile_first": group.tile_axis.global_tile_first,
        },
        "shards": [{
            "shard_ordinal": shard.shard_ordinal,
            "source_rank": shard.source_rank,
            "target_rank": shard.target_rank,
            "total_data_bytes": shard.total_data_bytes,
            "bytes_per_unit": shard.bytes_per_unit,
            "noc_path": list(shard.noc_path),
        } for shard in group.shards],
    }


def _endpoint_run_wire(run: PipelineEndpointRun) -> dict:
    return {
        "rank": run.rank,
        "group_ordinal": run.group_ordinal,
        "first_tile": run.first_tile,
        "tile_count": run.tile_count,
        "shard_ordinals": list(run.shard_ordinals),
        "first_json_id": run.first_json_id,
        "tile_stride": run.tile_stride,
        "shard_stride": run.shard_stride,
        "initial_parent_json_id": run.initial_parent_json_id,
        "legacy_context": {
            "request_id": run.legacy_context.request_id,
            "stage": run.legacy_context.stage,
            "generation": run.legacy_context.generation,
        },
        "legs": [{
            "endpoint": leg.endpoint,
            "json_offset": leg.json_offset,
            "tag_emission_ordinal": leg.tag_emission_ordinal,
            "tag_offset": leg.tag_offset,
            "tag_tile_stride": leg.tag_tile_stride,
            "tag_shard_stride": leg.tag_shard_stride,
        } for leg in run.legs],
    }


def _endpoint_dependency_run_wire(
        run: PipelineEndpointDependencyRun) -> dict:
    return {
        "rank": run.rank,
        "group_ordinal": run.group_ordinal,
        "first_tile": run.first_tile,
        "tile_count": run.tile_count,
        "shard_ordinals": list(run.shard_ordinals),
        "endpoint": run.endpoint,
        "dependency_kind": run.dependency_kind,
        "first_dependency_json_id": run.first_dependency_json_id,
        "dependency_tile_stride": run.dependency_tile_stride,
        "dependency_shard_stride": run.dependency_shard_stride,
    }


def _consume_completion_run_wire(
        run: PipelineConsumeCompletionRun) -> dict:
    return {
        "rank": run.rank,
        "group_ordinal": run.group_ordinal,
        "first_tile": run.first_tile,
        "tile_count": run.tile_count,
        "shard_ordinals": list(run.shard_ordinals),
        "dependency_kind": run.dependency_kind,
        "first_dependency_json_id": run.first_dependency_json_id,
        "dependency_tile_stride": run.dependency_tile_stride,
        "dependency_shard_stride": run.dependency_shard_stride,
    }


def _projection_share_wire(share: Optional[PipelineProjectionShare]):
    if share is None:
        return None
    return {
        "rule": share.rule,
        "total_units": share.total_units,
        "total_tiles": share.total_tiles,
        "share_index_first": share.share_index_first,
        "tile_bytes": share.tile_bytes,
    }


def _projection_wire(projection: PipelineProjection) -> dict:
    chains = []
    for chain in projection.chains:
        fill = chain.fill
        consume = chain.consume
        chains.append({
            "group_ordinal": chain.group_ordinal,
            "rank": chain.rank,
            "tile_count": chain.tile_count,
            "fill": {
                "leg_id_prefix": fill.leg_id_prefix,
                "first_index": fill.first_index,
                "fill_order_first": fill.fill_order_first,
                "fill_order_stride": fill.fill_order_stride,
                "completion_order_first": fill.completion_order_first,
                "completion_order_stride": fill.completion_order_stride,
                "route": list(fill.route),
                "bytes_per_tile": fill.bytes_per_tile,
                "latency_ns": fill.latency_ns,
                "start_deps": list(fill.start_deps),
                "gate_id": fill.gate_id,
                "gate_capacity": fill.gate_capacity,
            },
            "consume": {
                "leg_id_prefix": consume.leg_id_prefix,
                "first_index": consume.first_index,
                "consume_order_first": consume.consume_order_first,
                "consume_order_stride": consume.consume_order_stride,
                "completion_order_first": consume.completion_order_first,
                "completion_order_stride": consume.completion_order_stride,
                "kind": consume.kind,
                "route": list(consume.route),
                "group": consume.group,
                "latency_first_ns": consume.latency_first_ns,
                "latency_rest_ns": consume.latency_rest_ns,
                "consume_deps": list(consume.consume_deps),
                "consume_start_deps": list(consume.consume_start_deps),
                "gate_id": consume.gate_id,
                "compute_share": _projection_share_wire(consume.compute_share),
                "hbm_share": _projection_share_wire(consume.hbm_share),
            },
        })
    return {"version": projection.version, "chains": chains}


def _anchor_wire(anchor: PipelineAnchor) -> dict:
    selector = {"kind": anchor.selector.kind}
    for key in ("group_ordinal", "tile_ordinal", "shard_ordinal", "endpoint",
                "existing_gate", "tag_allocation_emission_ordinal", "tag_offset"):
        value = getattr(anchor.selector, key)
        if value is not None:
            selector[key] = value
    return {"role": anchor.role, "rank": anchor.rank,
            "binding": anchor.binding, "node_id": anchor.node_id,
            "selector": selector}
