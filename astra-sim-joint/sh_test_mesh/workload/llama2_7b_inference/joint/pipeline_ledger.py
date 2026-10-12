"""Strict protocol-v3 archive and protocol-v4 live pipeline consumers.

The executor owns mutable cursors.  This module accepts one immutable v3
archive or v4 delivery at a time, validates sparse changes against the last
accepted state, and keeps only current cursor/resource rows plus compact
descriptor indexes. It deliberately does not expand endpoint runs into
per-tile state.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
from collections import ChainMap, namedtuple
from collections.abc import Mapping as _RuntimeMapping
from bisect import bisect_left, bisect_right
from types import MappingProxyType
from typing import Any, Mapping, Optional

from joint.pipeline_descriptor import PipelineDescriptor, PipelineDescriptorError
from joint.work_ledger import (
    EVENT_PAYLOAD_WHITELIST,
    WorkLedgerError,
    _amount_number,
    _typed_v3_node_event_from_validated_scalars,
    amount_from_mapping,
    canonical_work_id,
    leg_work_id,
)


PRODUCER = "cpp_online"
SCHEMA_VERSION = 4
PROTOCOL_VERSION = 4
BRIDGE_PROTOCOL = "bridge-v4-20261011"
WORK_LEDGER_SCHEMA = "workledger-v4-20261011"
ARCHIVE_V3_SCHEMA_VERSION = 3
ARCHIVE_V3_PROTOCOL_VERSION = 3
ARCHIVE_V3_BRIDGE_PROTOCOL = "bridge-v3-20261009"
ARCHIVE_V3_WORK_LEDGER_SCHEMA = "workledger-v3-20261009"
DESCRIPTOR_VERSION = 1
U64_MAX = (1 << 64) - 1
I32_MAX = (1 << 31) - 1

_TOP_KEYS = frozenset({
    "producer", "schema_version", "protocol_version", "bridge_protocol",
    "work_ledger_schema", "epoch", "delivery_sequence", "base_watermark",
    "logical_seq_start", "logical_seq_end", "logical_event_count",
    "replay_token", "descriptor_version", "base_state_version",
    "result_state_version", "snapshot_kind", "tick", "descriptor_commits",
    "node_transitions", "pipeline_transition_runs", "cursor_patches",
    "active_resources", "optional_sha256",
})
_RUN_KEYS = frozenset({
    "first_logical_seq", "pattern_id", "repeat_count", "tile_first",
    "tile_stride", "rank_first", "rank_stride", "shard_first",
    "shard_stride", "transition_kind", "descriptor_segment_id",
    "endpoint_selector", "store_id_first", "store_id_stride",
    "legacy_context", "tick_first", "tick_stride", "resource_leg",
    "resource_quantity_prefix", "cursor_version_start",
    "cursor_version_end", "slot_patches", "credit_patches",
    "obligation_patches",
})
_TRANSITIONS = frozenset({
    "descriptor_commit", "tile_issue", "data_ready", "endpoint_done", "fill_ack_progress",
    "fill_ack_done", "consume_start", "consume_done",
    "release_ack_progress", "release_ack_done", "credit_return",
    "service_done", "request_close", "settle_done", "descriptor_retire",
})
_RESOURCE_LEGS = frozenset({
    None, "endpoint", "body_marker", "noc", "hbm:COMM_READ",
    "hbm:COMM_WRITE",
})
_SLOT_KEYS = frozenset({
    "buffer_domain_id", "role", "rank", "shard_ordinal", "slot_index",
    "occupied", "owner", "consumed", "target_shards_to_consume",
    "consumed_target_shards",
})
_OWNER_KEYS = frozenset({
    "segment_id", "pipeline_generation", "stage_epoch",
    "staging_allocation_id", "group_ordinal", "tile_ordinal",
})
_CREDIT_KEYS = frozenset({
    "buffer_domain_id", "source_rank", "shard_ordinal", "capacity",
    "available", "outstanding_ack_count",
})
_OBLIGATION_KEYS = frozenset({
    "obligation_id", "segment_id", "rank", "first_store_id",
    "remaining_endpoint_count", "state",
})
_CURSOR_RUN_KEYS = frozenset({
    "rank", "group_ordinal", "first_tile", "tile_count",
    "completed_endpoint_count", "cancelled_endpoint_count",
    "remaining_endpoint_count", "in_flight",
    "next_store_id", "tile_offset", "shard_index", "leg_index",
    "initial_parent_store_id", "completed_out_of_order", "active_endpoints",
})
_OOO_KEYS = frozenset({"first_ordinal", "end_ordinal_exclusive"})
# Hoisted per-row key sets shared by the offline decoder and the fused
# streaming validator (identical membership; the literal was previously
# rebuilt once per node row).
_NODE_ROW_KEYS = frozenset(
    {"logical_seq", "tick", "kind", "work_id", "payload"})
_NODE_EVENT_KINDS = frozenset({
    "commit", "issue", "progress", "completion", "ack", "release"})
_ACTIVE_ENDPOINT_KEYS = frozenset({
    "store_id", "tile_ordinal", "shard_ordinal", "endpoint", "transfer_id",
    "flow_id", "network_done", "hbm_done", "owner",
})
_RESOURCE_COMMON = frozenset({
    "resource_type", "work_id", "request_id", "stage", "generation", "rank",
    "node_id", "operation", "leg", "state", "total", "remaining",
    "progress_source", "service_unit", "route", "as_of_tick", "issued_tick",
    "descriptor_segment_id", "endpoint_selector",
})
_RESOURCE_EXTRAS = {
    "noc_flow": frozenset({"flow_id", "current_rate_Bpns", "last_rate_update_time"}),
    "hbm_job": frozenset({"kind", "start_tick"}),
    "pool_job": frozenset({"port_index", "issue_seq", "latency_ready_ns"}),
}


class PipelineLedgerError(ValueError):
    """Malformed v3 pipeline envelope or inconsistent state transition."""


def _mapping(value: Any, path: str, keys: frozenset[str]) -> Mapping[str, Any]:
    value_type = type(value)
    if (value_type is not dict and value_type is not MappingProxyType
            and not issubclass(value_type, _RuntimeMapping)):
        raise PipelineLedgerError(f"{path} must be an object")
    missing = keys - set(value)
    extra = set(value) - keys
    if missing or extra:
        raise PipelineLedgerError(
            f"{path} keys invalid (missing={sorted(missing)}, unknown={sorted(extra)})")
    return value


def _array(value: Any, path: str) -> tuple:
    if not isinstance(value, (list, tuple)):
        raise PipelineLedgerError(f"{path} must be an array")
    return tuple(value)


def _uint(value: Any, path: str, maximum: int = U64_MAX) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise PipelineLedgerError(f"{path} must be uint in [0,{maximum}] (bool rejected)")
    return value


def _rank(value: Any, path: str) -> int:
    return _uint(value, path, I32_MAX)


def _finite_nonnegative(value: Any, path: str) -> int | float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise PipelineLedgerError(f"{path} must be finite nonnegative int/float (bool rejected)")
    return value


def _string(value: Any, path: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value):
        raise PipelineLedgerError(f"{path} must be {'a string' if empty else 'a non-empty string'}")
    return value


def _boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise PipelineLedgerError(f"{path} must be bool")
    return value


def _nullable_uint(value: Any, path: str) -> Optional[int]:
    return None if value is None else _uint(value, path)


def _canonical_envelope_bytes(value: Any) -> bytes:
    """One exact, bounded replay representation (JSON sorts keys and keeps
    integer/float spellings distinct, unlike Python equality)."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PipelineLedgerError(
            f"envelope cannot be represented as canonical JSON: {exc}") from exc


def _typed_equal(left: Any, right: Any) -> bool:
    """Compare parsed protocol values without Python's int/float coercion."""
    left_type, right_type = type(left), type(right)
    if ((left_type is dict or left_type is MappingProxyType)
            and (right_type is dict or right_type is MappingProxyType)):
        return (left.keys() == right.keys()
                and all(_typed_equal(left[key], right[key]) for key in left))
    if (issubclass(left_type, _RuntimeMapping)
            and issubclass(right_type, _RuntimeMapping)):
        return (left.keys() == right.keys()
                and all(_typed_equal(left[key], right[key]) for key in left))
    if (isinstance(left, (tuple, list))
            and isinstance(right, (tuple, list))):
        return (len(left) == len(right)
                and all(_typed_equal(a, b) for a, b in zip(left, right)))
    if type(left) is not type(right):
        return False
    return left == right


def _freeze_dto(value: Any) -> Any:
    value_type = type(value)
    if value_type is dict or value_type is MappingProxyType:
        return MappingProxyType({key: _freeze_dto(item) for key, item in value.items()})
    if issubclass(value_type, _RuntimeMapping):
        return MappingProxyType({key: _freeze_dto(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_dto(item) for item in value)
    return value


def _plain_obligation_row_for_postwrite(value: Any) -> bool:
    """Whether row strings/keys cannot make typed equality stateful."""
    if value is None:
        return True
    if type(value) not in (dict, MappingProxyType):
        return False
    if any(type(key) is not str for key in value):
        return False
    try:
        string_fields = (
            value["obligation_id"], value["segment_id"], value["state"])
    except KeyError:
        return False
    return all(type(field) is str for field in string_fields)


def _amount(value: Any, path: str) -> Mapping[str, Any]:
    row = _mapping(value, path, frozenset({"ops", "bytes", "latency_ns"}))
    for dimension in ("ops", "bytes", "latency_ns"):
        if row[dimension] is not None:
            _finite_nonnegative(row[dimension], f"{path}.{dimension}")
    return row


def _canonical_rows(rows, key, path):
    values = _array(rows, path)
    previous = None
    for row in values:
        current = key(row)
        if previous is not None and current <= previous:
            raise PipelineLedgerError(
                f"{path} must be sorted and duplicate-free")
        previous = current
    return values


def _parse_row_patches(value: Any, path: str, parse_rows,
                       key_of, sort_key):
    """Parse sparse before/after row patches into detached typed values."""
    parsed = []
    previous_key = None
    for index, raw in enumerate(_array(value, path)):
        p = f"{path}[{index}]"
        patch = _mapping(raw, p, frozenset({"before", "after"}))
        before_raw, after_raw = patch["before"], patch["after"]
        if before_raw is None and after_raw is None:
            raise PipelineLedgerError(f"{p} cannot have both before and after null")
        before = (None if before_raw is None else
                  parse_rows((before_raw,), p + ".before")[0])
        after = (None if after_raw is None else
                 parse_rows((after_raw,), p + ".after")[0])
        if before is not None and after is not None:
            before_key, after_key = key_of(before), key_of(after)
            if before_key != after_key:
                raise PipelineLedgerError(f"{p} changes a row's stable key")
            if _typed_equal(before, after):
                raise PipelineLedgerError(f"{p} is a no-op patch")
            key = before_key
        else:
            key = key_of(before if before is not None else after)
        ordered_key = sort_key(key)
        if previous_key is not None and ordered_key <= previous_key:
            raise PipelineLedgerError(
                f"{path} must be sorted and duplicate-free")
        previous_key = ordered_key
        parsed.append((key, before, after))
    return tuple(parsed)


_TARGET_DELTA_KEYS = frozenset({
    "form", "key", "before_slot_generation", "predecessor_logical_seq",
    "predecessor_consumed_count", "operations",
})
_TARGET_KEY_KEYS = frozenset({
    "buffer_domain_id", "role", "rank", "shard_ordinal", "slot_index",
})


def _parse_v4_slot_patches(value: Any, path: str,
                           descriptors: Mapping[str, Any],
                           selector_indexes: Optional[Mapping[str, Any]] = None
                           ) -> tuple:
    """Parse the v4 source-row / target-delta tagged slot-patch union."""
    parsed = []
    previous_key = None
    for index, raw in enumerate(_array(value, path)):
        p = f"{path}[{index}]"
        if (issubclass(type(raw), _RuntimeMapping)
                and raw.get("form") == "target_delta"):
            delta = _mapping(raw, p, _TARGET_DELTA_KEYS)
            if delta["form"] != "target_delta":
                raise PipelineLedgerError(f"{p}.form is unknown")
            key_row = _mapping(delta["key"], p + ".key", _TARGET_KEY_KEYS)
            domain = _string(key_row["buffer_domain_id"], p + ".key.buffer_domain_id")
            role = key_row["role"]
            rank = _rank(key_row["rank"], p + ".key.rank")
            shard = _nullable_uint(key_row["shard_ordinal"],
                                   p + ".key.shard_ordinal")
            slot_index = _uint(key_row["slot_index"], p + ".key.slot_index", 1)
            if role != "target" or shard is not None:
                raise PipelineLedgerError(
                    f"{p}.key must identify a target slot with null shard")
            key = (domain, role, rank, shard, slot_index)
            before_generation = _uint(
                delta["before_slot_generation"],
                p + ".before_slot_generation")
            predecessor_seq = _uint(
                delta["predecessor_logical_seq"],
                p + ".predecessor_logical_seq")
            predecessor_count = _uint(
                delta["predecessor_consumed_count"],
                p + ".predecessor_consumed_count")
            operations = []
            for op_index, raw_op in enumerate(_array(
                    delta["operations"], p + ".operations")):
                q = f"{p}.operations[{op_index}]"
                if not issubclass(type(raw_op), _RuntimeMapping):
                    raise PipelineLedgerError(f"{q} must be an object")
                op_kind = _string(raw_op.get("op"), q + ".op")
                generation = _uint(raw_op.get("slot_generation"),
                                   q + ".slot_generation")
                if op_kind == "initialize":
                    allowed = frozenset({"op", "slot_generation"})
                    _mapping(raw_op, q, allowed)
                    operation = {"op": op_kind, "slot_generation": generation}
                elif op_kind == "claim":
                    allowed = frozenset({
                        "op", "slot_generation", "owner", "target_shards"})
                    op_row = _mapping(raw_op, q, allowed)
                    owner_raw = _mapping(op_row["owner"], q + ".owner", _OWNER_KEYS)
                    owner = dict(owner_raw)
                    owner["segment_id"] = _string(
                        owner_raw["segment_id"], q + ".owner.segment_id")
                    for field in ("pipeline_generation", "stage_epoch",
                                  "staging_allocation_id", "group_ordinal",
                                  "tile_ordinal"):
                        owner[field] = _uint(
                            owner_raw[field], q + ".owner." + field)
                    target_shards = tuple(
                        _uint(item, q + ".target_shards[]")
                        for item in _array(op_row["target_shards"],
                                           q + ".target_shards"))
                    if any(left >= right for left, right in
                           zip(target_shards, target_shards[1:])):
                        raise PipelineLedgerError(
                            f"{q}.target_shards must be sorted unique")
                    operation = {
                        "op": op_kind, "slot_generation": generation,
                        "owner": owner,
                        "target_shards": target_shards,
                    }
                elif op_kind == "consume":
                    allowed = frozenset({
                        "op", "slot_generation", "cause", "add_shards",
                        "cleared"})
                    op_row = _mapping(raw_op, q, allowed)
                    cause_row = _mapping(op_row["cause"], q + ".cause",
                        frozenset({"terminal_rank", "terminal_store_id",
                                   "consume_run_index"}))
                    cause = {
                        "terminal_rank": _rank(
                            cause_row["terminal_rank"],
                            q + ".cause.terminal_rank"),
                        "terminal_store_id": _uint(
                            cause_row["terminal_store_id"],
                            q + ".cause.terminal_store_id"),
                        "consume_run_index": _uint(
                            cause_row["consume_run_index"],
                            q + ".cause.consume_run_index"),
                    }
                    additions = tuple(
                        _uint(item, q + ".add_shards[]")
                        for item in _array(op_row["add_shards"],
                                           q + ".add_shards"))
                    if any(left >= right for left, right in
                           zip(additions, additions[1:])):
                        raise PipelineLedgerError(
                            f"{q}.add_shards must be sorted unique")
                    cleared = _boolean(op_row["cleared"], q + ".cleared")
                    operation = {
                        "op": op_kind, "slot_generation": generation,
                        "cause": cause, "add_shards": additions,
                        "cleared": cleared,
                    }
                elif op_kind == "request_close_clear":
                    _mapping(raw_op, q, frozenset({"op", "slot_generation"}))
                    operation = {"op": op_kind,
                                 "slot_generation": generation}
                elif op_kind == "erase":
                    _mapping(raw_op, q, frozenset({"op", "slot_generation"}))
                    operation = {"op": op_kind,
                                 "slot_generation": generation}
                else:
                    raise PipelineLedgerError(f"{q}.op is unknown")
                operations.append(operation)
            if not operations:
                raise PipelineLedgerError(f"{p}.operations must be nonempty")
            item = {
                "form": "target_delta", "key": key,
                "before_slot_generation": before_generation,
                "predecessor_logical_seq": predecessor_seq,
                "predecessor_consumed_count": predecessor_count,
                "operations": tuple(operations),
            }
            sort_key = (key[0], key[1], key[2], -1, key[4])
            parsed.append((sort_key, item))
        else:
            before_after = _parse_row_patches(
                (raw,), p,
                lambda values, q: _parse_slot_rows(
                    values, q, descriptors, selector_indexes),
                lambda row: (row["buffer_domain_id"], row["role"],
                             row["rank"], row["shard_ordinal"],
                             row["slot_index"]),
                lambda key: (key[0], key[1], key[2],
                             -1 if key[3] is None else key[3], key[4]))[0]
            key, before, after = before_after
            if ((before is not None and before["role"] != "source")
                    or (after is not None and after["role"] != "source")):
                raise PipelineLedgerError(
                    f"{p} v4 target slots must use target_delta form")
            sort_key = (key[0], key[1], key[2],
                        -1 if key[3] is None else key[3], key[4])
            parsed.append((sort_key, ("source", key, before, after)))
        current_key = parsed[-1][0]
        if previous_key is not None and current_key <= previous_key:
            raise PipelineLedgerError(
                f"{path} must be sorted and duplicate-free")
        previous_key = current_key
    return tuple(item for _sort, item in parsed)


@dataclass(frozen=True)
class PipelineLedgerReceipt:
    snapshot_kind: str
    delivery_sequence: int
    logical_event_count: int
    base_watermark: int
    event_watermark: int
    base_state_version: int
    state_version: int
    replayed: bool = False


@dataclass(frozen=True)
class _ParsedEnvelope:
    canonical_bytes: bytes
    replay_token: str
    producer: str
    epoch: str
    delivery_sequence: int
    snapshot_kind: str
    tick: int
    base_watermark: int
    seq_start: int
    seq_end: int
    event_count: int
    base_version: int
    result_version: int
    protocol_version: int
    descriptors: tuple
    selector_index_additions: Mapping[str, Any]
    dependency_index_additions: Mapping[str, Any]
    node_events: tuple
    pipeline_runs: tuple
    cursor_patches: tuple
    resources: tuple


@dataclass(frozen=True)
class PreparedPipelineUpdate:
    parsed: _ParsedEnvelope
    descriptor_additions: Mapping[str, PipelineDescriptor]
    descriptor_state_updates: Mapping[str, Any]
    descriptor_retirements: tuple
    cursor_changes: Mapping[Any, Any]
    cursor_version_updates: Mapping[str, int]
    cursor_transfer_changes: Mapping[Any, Any]
    active_endpoint_deltas: Mapping[str, int]
    resources: tuple
    slot_changes: Mapping[Any, Any]
    target_slot_lineage_changes: Mapping[Any, Any]
    target_ready_changes: Mapping[Any, Any]
    terminal_provenance_updates: Mapping[int, Any]
    credit_changes: Mapping[Any, Any]
    obligation_changes: Mapping[Any, Any]
    aggregate_deltas: Mapping[str, Mapping[Any, int]]
    selector_index_additions: Mapping[str, Any]
    dependency_index_additions: Mapping[str, Any]
    dependency_rank_index_additions: Mapping[int, Any]
    changed: bool
    replayed: bool = False


_ROW_TOMBSTONE = object()


class _RowOverlay:
    """Touched-row transaction overlay over one accepted ledger index."""

    __slots__ = ("base", "writes", "deleted")

    def __init__(self, base):
        self.base = base
        self.writes = {}
        self.deleted = set()

    def get(self, key, default=None):
        if key in self.writes:
            return self.writes[key]
        if key in self.deleted:
            return default
        return self.base.get(key, default)

    def has(self, key):
        return key in self.writes or (key not in self.deleted and key in self.base)

    def set(self, key, value):
        self.deleted.discard(key)
        self.writes[key] = value

    def delete(self, key):
        self.writes.pop(key, None)
        self.deleted.add(key)

    def changes(self):
        return {**{key: _ROW_TOMBSTONE for key in self.deleted}, **self.writes}


class _CountOverlay:
    """Sparse integer aggregate deltas, staged with the row overlay."""

    __slots__ = ("base", "deltas")

    def __init__(self, base):
        self.base = base
        self.deltas = {}

    def get(self, key):
        return self.base.get(key, 0) + self.deltas.get(key, 0)

    def add(self, key, amount):
        if amount:
            self.deltas[key] = self.deltas.get(key, 0) + amount


def _slot_key(row):
    return (row["buffer_domain_id"], row["role"], row["rank"],
            row["shard_ordinal"], row["slot_index"])


def _global_tile_ordinal(descriptor, group_ordinal, tile_ordinal,
                         selector_index=None):
    group = (None if selector_index is None else
             selector_index.groups_by_ordinal.get(group_ordinal))
    if group is None:
        group = next((item for item in descriptor.groups
                      if item.group_ordinal == group_ordinal), None)
    if (group is None or tile_ordinal < 1
            or tile_ordinal > group.tile_axis.tile_count):
        raise PipelineLedgerError(
            "slot owner points outside descriptor group/tile range")
    return group.tile_axis.global_tile_first + tile_ordinal - 1


def _credit_key(row):
    return (row["buffer_domain_id"], row["source_rank"], row["shard_ordinal"])


def _obligation_key(row):
    return row["obligation_id"]


def _target_owner_tag(owner):
    if owner is None:
        return None
    return tuple(owner[field] for field in (
        "segment_id", "pipeline_generation", "stage_epoch",
        "staging_allocation_id", "group_ordinal", "tile_ordinal"))


@dataclass(frozen=True)
class _DescriptorSelectorIndex:
    """Run and selector indexes bounded by descriptor runs/shards/legs."""

    groups_by_ordinal: Mapping[Any, Any]
    run_intervals: Mapping[Any, tuple]
    selector_intervals: Mapping[Any, tuple]
    consume_intervals_by_target: Mapping[Any, tuple]


@dataclass(frozen=True)
class _DescriptorDependencyIndex:
    intervals_by_prefix: Mapping[str, tuple]
    intervals_by_rank: Mapping[int, tuple]


@dataclass(frozen=True)
class _DependencyIntervalNode:
    row: tuple
    priority: int
    max_last: int
    left: Any = None
    right: Any = None


@dataclass(frozen=True)
class _CompletedIdIntervalNode:
    first: int
    last: int
    priority: int
    left: Any = None
    right: Any = None


def _completed_interval_priority(value):
    # SplitMix64 disperses sequential store IDs while keeping the tree shape
    # deterministic and independent of replay-visible data. This is an
    # expected/observed-height treap, not an adversarial worst-case bound.
    mixed = (value + 0x9E3779B97F4A7C15) & U64_MAX
    mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & U64_MAX
    mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & U64_MAX
    return (mixed ^ (mixed >> 31)) & U64_MAX


def _completed_interval_node(first, last, priority, left=None, right=None):
    return _CompletedIdIntervalNode(first, last, priority, left, right)


def _completed_interval_heap_key(node):
    return (node.priority, node.first)


def _completed_interval_rotate_right(root):
    left = root.left
    new_right = _completed_interval_node(
        root.first, root.last, root.priority, left.right, root.right)
    return _completed_interval_node(
        left.first, left.last, left.priority, left.left, new_right)


def _completed_interval_rotate_left(root):
    right = root.right
    new_left = _completed_interval_node(
        root.first, root.last, root.priority, root.left, right.left)
    return _completed_interval_node(
        right.first, right.last, right.priority, new_left, right.right)


def _completed_interval_insert(root, first, last, work=None):
    priority = _completed_interval_priority(first)
    if root is None:
        return _completed_interval_node(first, last, priority)
    if work is not None:
        work["tree_update_node_visits"] = (
            work.get("tree_update_node_visits", 0) + 1)
    if first == root.first:
        if last != root.last:
            raise PipelineLedgerError(
                "completed terminal interval key was reused inconsistently")
        return root
    if first < root.first:
        left = _completed_interval_insert(root.left, first, last, work)
        result = _completed_interval_node(
            root.first, root.last, root.priority, left, root.right)
        return (_completed_interval_rotate_right(result)
                if _completed_interval_heap_key(left)
                < _completed_interval_heap_key(result) else result)
    right = _completed_interval_insert(root.right, first, last, work)
    result = _completed_interval_node(
        root.first, root.last, root.priority, root.left, right)
    return (_completed_interval_rotate_left(result)
            if _completed_interval_heap_key(right)
            < _completed_interval_heap_key(result) else result)


def _completed_interval_merge(left, right, work=None):
    if left is None:
        return right
    if right is None:
        return left
    if work is not None:
        work["tree_update_node_visits"] = (
            work.get("tree_update_node_visits", 0) + 1)
    if _completed_interval_heap_key(left) < _completed_interval_heap_key(right):
        merged = _completed_interval_merge(left.right, right, work)
        return _completed_interval_node(
            left.first, left.last, left.priority, left.left, merged)
    merged = _completed_interval_merge(left, right.left, work)
    return _completed_interval_node(
        right.first, right.last, right.priority, merged, right.right)


def _completed_interval_delete(root, first, work=None):
    if root is None:
        return None
    if work is not None:
        work["tree_update_node_visits"] = (
            work.get("tree_update_node_visits", 0) + 1)
    if first == root.first:
        return _completed_interval_merge(root.left, root.right, work)
    if first < root.first:
        left = _completed_interval_delete(root.left, first, work)
        return _completed_interval_node(
            root.first, root.last, root.priority, left, root.right)
    right = _completed_interval_delete(root.right, first, work)
    return _completed_interval_node(
        root.first, root.last, root.priority, root.left, right)


def _completed_interval_neighbors(root, value, work=None):
    predecessor = successor = None
    current = root
    while current is not None:
        if work is not None:
            work["tree_query_node_visits"] = (
                work.get("tree_query_node_visits", 0) + 1)
        if current.first < value:
            predecessor = current
            current = current.right
        elif current.first > value:
            successor = current
            current = current.left
        else:
            return predecessor, current
    return predecessor, successor


def _completed_interval_contains(root, value, work=None):
    current = root
    while current is not None:
        if work is not None:
            work["tree_query_node_visits"] = (
                work.get("tree_query_node_visits", 0) + 1)
        if value < current.first:
            current = current.left
        elif value > current.last:
            current = current.right
        else:
            return True
    return False


def _completed_interval_add(root, value, work=None):
    if _completed_interval_contains(root, value, work):
        return root
    predecessor, successor = _completed_interval_neighbors(root, value, work)
    first = last = value
    if predecessor is not None and predecessor.last + 1 == value:
        first = predecessor.first
        root = _completed_interval_delete(root, predecessor.first, work)
    if successor is not None and successor.first == value + 1:
        last = successor.last
        root = _completed_interval_delete(root, successor.first, work)
    return _completed_interval_insert(root, first, last, work)


def _completed_interval_build_from_half_open(ranges, work=None):
    """Build a canonical completed-range treap in O(H) from sorted rows.

    Input/output ranges are half-open. Tree nodes retain inclusive endpoints,
    like the shared completed-ID provenance treap. Adjacent wire ranges are
    normalized together, matching the historical per-transition merge pass.
    Returns ``(root, completed_out_of_order_count)``.
    """
    canonical = []
    completed_count = 0
    previous_end = None
    for first, end in ranges:
        if work is not None:
            work["seed_interval_rows"] = (
                work.get("seed_interval_rows", 0) + 1)
        if end <= first or (previous_end is not None and first < previous_end):
            raise PipelineLedgerError(
                "completed endpoint ranges are empty, overlapping, or unsorted")
        completed_count += end - first
        if canonical and first == canonical[-1][1]:
            canonical[-1] = (canonical[-1][0], end)
        else:
            canonical.append((first, end))
        previous_end = end

    size = len(canonical)
    if size == 0:
        return None, completed_count

    priorities = [
        _completed_interval_priority(first) for first, _end in canonical]
    left = [-1] * size
    right = [-1] * size
    stack = []
    for index, (first, _end) in enumerate(canonical):
        last_popped = -1
        key = (priorities[index], first)
        while stack and key < (priorities[stack[-1]], canonical[stack[-1]][0]):
            last_popped = stack.pop()
            if work is not None:
                work["seed_stack_pop_steps"] = (
                    work.get("seed_stack_pop_steps", 0) + 1)
        if stack:
            right[stack[-1]] = index
        if last_popped >= 0:
            left[index] = last_popped
        stack.append(index)
        if work is not None:
            work["seed_build_steps"] = (
                work.get("seed_build_steps", 0) + 1)

    root_index = stack[0]
    postorder = []
    pending = [(root_index, False)]
    while pending:
        index, visited = pending.pop()
        if work is not None:
            work["seed_postorder_visits"] = (
                work.get("seed_postorder_visits", 0) + 1)
        if visited:
            postorder.append(index)
            continue
        pending.append((index, True))
        if right[index] >= 0:
            pending.append((right[index], False))
        if left[index] >= 0:
            pending.append((left[index], False))

    nodes = [None] * size
    for index in postorder:
        first, end = canonical[index]
        nodes[index] = _completed_interval_node(
            first, end - 1, priorities[index],
            nodes[left[index]] if left[index] >= 0 else None,
            nodes[right[index]] if right[index] >= 0 else None)
        if work is not None:
            work["seed_tree_nodes"] = work.get("seed_tree_nodes", 0) + 1
    return nodes[root_index], completed_count


def _completed_interval_add_range(root, completed_prefix, total_count,
                                   first, end, work=None):
    """Return a persistent root/prefix after adding one exact half-open range.

    Only the predecessor, successor, and adjacent successor chain can overlap
    or merge. The caller publishes the returned pair only after this function
    succeeds, so a rejected range or allocation error leaves its prior root
    intact.
    """
    if work is not None:
        work["range_updates"] = work.get("range_updates", 0) + 1
    if (first < completed_prefix or end <= first or end > total_count):
        raise PipelineLedgerError(
            "completed range is duplicate or outside its cursor extent")

    predecessor, successor = _completed_interval_neighbors(root, first, work)
    if predecessor is not None and predecessor.last >= first:
        raise PipelineLedgerError("completed range overlaps its predecessor")
    if successor is not None and successor.first < end:
        raise PipelineLedgerError("completed range overlaps its successor")

    merged_first = first
    merged_end = end
    if predecessor is not None and predecessor.last + 1 == first:
        merged_first = predecessor.first
        root = _completed_interval_delete(root, predecessor.first, work)

    while True:
        _prior, next_interval = _completed_interval_neighbors(
            root, merged_end, work)
        if (next_interval is None or
                next_interval.first != merged_end):
            break
        merged_end = next_interval.last + 1
        root = _completed_interval_delete(
            root, next_interval.first, work)

    if merged_first == completed_prefix:
        completed_prefix = merged_end
    else:
        root = _completed_interval_insert(
            root, merged_first, merged_end - 1, work)
    # The legacy sorted-list merge folded every completed range beginning at
    # the prefix after each completion, including a canonicalizable interval
    # already present in the seeded cursor row.
    while True:
        _prior, next_interval = _completed_interval_neighbors(
            root, completed_prefix, work)
        if (next_interval is None or
                next_interval.first != completed_prefix):
            break
        completed_prefix = next_interval.last + 1
        root = _completed_interval_delete(
            root, next_interval.first, work)
    return root, completed_prefix


def _completed_interval_export_half_open(root, work=None):
    """Canonical in-order export, called once when a final cursor patch is checked."""
    result = []
    stack = []
    current = root
    while current is not None or stack:
        while current is not None:
            stack.append(current)
            current = current.left
        current = stack.pop()
        if work is not None:
            work["export_node_visits"] = (
                work.get("export_node_visits", 0) + 1)
        result.append((current.first, current.last + 1))
        current = current.right
    return tuple(result)


def _dependency_interval_key(row):
    return tuple(row)


def _dependency_interval_node(row, priority, left=None, right=None):
    return _DependencyIntervalNode(
        row=row, priority=priority,
        max_last=max(row[1],
                     left.max_last if left is not None else 0,
                     right.max_last if right is not None else 0),
        left=left, right=right)


def _dependency_rotate_right(root):
    left = root.left
    new_right = _dependency_interval_node(
        root.row, root.priority, left.right, root.right)
    return _dependency_interval_node(
        left.row, left.priority, left.left, new_right)


def _dependency_rotate_left(root):
    right = root.right
    new_left = _dependency_interval_node(
        root.row, root.priority, root.left, right.left)
    return _dependency_interval_node(
        right.row, right.priority, new_left, right.right)


def _dependency_interval_priority(row):
    # Treap priorities affect lookup shape only. Python's process-salted hash
    # is suitable here; no priority escapes into the replay representation.
    return hash(_dependency_interval_key(row)) & U64_MAX


def _dependency_interval_insert(root, row):
    """Persistent treap insertion; copies only one root-to-leaf path."""
    priority = _dependency_interval_priority(row)
    if root is None:
        return _dependency_interval_node(row, priority)
    key, current = _dependency_interval_key(row), _dependency_interval_key(root.row)
    if key == current:
        return root
    if key < current:
        left = _dependency_interval_insert(root.left, row)
        result = _dependency_interval_node(
            root.row, root.priority, left, root.right)
        return (_dependency_rotate_right(result)
                if left.priority < result.priority else result)
    right = _dependency_interval_insert(root.right, row)
    result = _dependency_interval_node(
        root.row, root.priority, root.left, right)
    return (_dependency_rotate_left(result)
            if right.priority < result.priority else result)


def _build_descriptor_selector_index(descriptor: PipelineDescriptor):
    run_rows = {}
    selector_rows = {}
    consume_rows = {}
    for run_index, run in enumerate(descriptor.endpoint_runs):
        start, end = run.first_tile, run.first_tile + run.tile_count
        run_rows.setdefault((run.group_ordinal, run.rank), []).append(
            (start, end, run_index))
        for shard in run.shard_ordinals:
            for leg_index, leg in enumerate(run.legs):
                selector_rows.setdefault(
                    (run.group_ordinal, run.rank, shard, leg.endpoint),
                    []).append((start, end, run_index, leg_index))
    for completion_index, completion in enumerate(
            descriptor.consume_completion_runs):
        start = completion.first_tile
        end = start + completion.tile_count
        for shard_position, shard in enumerate(completion.shard_ordinals):
            consume_rows.setdefault(
                (completion.rank, completion.group_ordinal, shard), []).append(
                    (start, end, completion_index, shard_position))

    def freeze(rows):
        result = {}
        for key, intervals in rows.items():
            intervals.sort()
            result[key] = (tuple(item[0] for item in intervals),
                           tuple(intervals))
        return MappingProxyType(result)

    return _DescriptorSelectorIndex(
        MappingProxyType({group.group_ordinal: group
                          for group in descriptor.groups}),
        freeze(run_rows), freeze(selector_rows), freeze(consume_rows))


def _build_descriptor_dependency_index(descriptor: PipelineDescriptor):
    rows = {}
    rank_rows = {}
    segment = descriptor.identity.segment_id
    for run_index, run in enumerate(descriptor.endpoint_runs):
        context = run.legacy_context
        prefix = (f"{context.request_id}#{context.stage}:"
                  f"{context.generation}:r{run.rank}n")
        stride = run.tile_stride
        ordinal_tile_stride = len(run.shard_ordinals) * len(run.legs)
        for shard_index, shard in enumerate(run.shard_ordinals):
            for leg_index, leg in enumerate(run.legs):
                first = descriptor.endpoint_id(
                    run_index, run.first_tile, shard, leg_index) + 1
                last = first + (run.tile_count - 1) * stride
                interval = (
                    first, last, segment, run.rank, stride, first % stride,
                    run_index, shard_index * len(run.legs) + leg_index,
                    ordinal_tile_stride)
                rows.setdefault(prefix, []).append(interval)
                rank_rows.setdefault(run.rank, []).append(interval)
    return _DescriptorDependencyIndex(MappingProxyType({
        prefix: tuple(sorted(intervals))
        for prefix, intervals in rows.items()}), MappingProxyType({
            rank: tuple(sorted(intervals))
            for rank, intervals in rank_rows.items()}))


def _dependency_interval_matches(root, node_id, matches):
    if root is None or root.max_last < node_id:
        return
    left = root.left
    if left is not None and left.max_last >= node_id:
        _dependency_interval_matches(left, node_id, matches)
    first, last, segment, rank, stride, residue = root.row[:6]
    if first <= node_id <= last and node_id % stride == residue:
        matches.add(root.row)
    right = root.right
    if first <= node_id and right is not None and right.max_last >= node_id:
        _dependency_interval_matches(right, node_id, matches)


def _dependency_node_parts(dependency):
    if not isinstance(dependency, str):
        return None
    try:
        prefix, node_text = dependency.rsplit("n", 1)
        node_id = int(node_text)
    except (ValueError, TypeError):
        return None
    # ``r0n`` is part of the canonical work-id prefix and the dependency
    # index is keyed by that full prefix. Keep the delimiter attached after
    # splitting the numeric NodeStore id.
    prefix += "n"
    if node_id < 1 or str(node_id) != node_text:
        return None
    return prefix, node_id


def _dependency_matches(dependency, *indexes):
    parts = _dependency_node_parts(dependency)
    if parts is None:
        return ()
    prefix, node_id = parts
    matches = set()
    for index in indexes:
        root = index.intervals_by_prefix.get(prefix) if hasattr(
            index, "intervals_by_prefix") else index.get(prefix)
        _dependency_interval_matches(root, node_id, matches)
    return tuple(sorted(matches))


def _dependency_segment(dependency, *indexes):
    matches = _dependency_matches(dependency, *indexes)
    segments = {row[2] for row in matches}
    if len(segments) > 1:
        raise PipelineLedgerError(
            "ordinary dependency maps to multiple pipeline endpoints")
    return next(iter(segments)) if segments else None


def _dependency_reference(dependency, *indexes):
    parts = _dependency_node_parts(dependency)
    if parts is None:
        return None
    _prefix, node_id = parts
    matches = _dependency_matches(dependency, *indexes)
    segments = {row[2] for row in matches}
    if len(segments) > 1:
        raise PipelineLedgerError(
            "ordinary dependency maps to multiple pipeline endpoints")
    if not matches:
        return None
    # A NodeStore id can be represented by more than one affine endpoint
    # selector. The exact logical id and owner rank remain the dependency
    # identity; endpoint_done supplies the exact cursor selector proof.
    row = matches[0]
    return row[2], row[3], node_id


def _dependency_segments_for_rank_node(rank, node_id, indexes):
    root = indexes.get(rank)
    matches = set()
    _dependency_interval_matches(root, node_id, matches)
    return {row[2] for row in matches}


def _descriptor_run_at(indexes, group, rank, tile):
    found = indexes.run_intervals.get((group, rank))
    if found is None:
        return None
    starts, intervals = found
    position = bisect_right(starts, tile) - 1
    if position < 0:
        return None
    start, end, run_index = intervals[position]
    return run_index if start <= tile < end else None


def _descriptor_selector_at(indexes, group, rank, tile, shard, endpoint):
    found = indexes.selector_intervals.get((group, rank, shard, endpoint))
    if found is None:
        return None
    starts, intervals = found
    position = bisect_right(starts, tile) - 1
    if position < 0:
        return None
    start, end, run_index, leg_index = intervals[position]
    return ((run_index, leg_index) if start <= tile < end else None)


def _consume_completion_at(indexes, rank, group, tile, shard):
    """Find the one immutable terminal run covering a target owner row."""
    found = indexes.consume_intervals_by_target.get((rank, group, shard))
    if found is None:
        return None
    starts, intervals = found
    position = bisect_right(starts, tile) - 1
    if position < 0:
        return None
    start, end, run_index, shard_position = intervals[position]
    return ((run_index, shard_position)
            if start <= tile < end else None)


def _run_endpoint_total(run):
    return run.tile_count * len(run.shard_ordinals) * len(run.legs)


def _cursor_position_ordinal(run_row, run):
    return (run_row["tile_offset"] * len(run.shard_ordinals)
            * len(run.legs)
            + run_row["shard_index"] * len(run.legs)
            + run_row["leg_index"])


def _completed_range_contains_half_open_sorted(ranges, ordinal):
    """Test one ordinal against sorted disjoint half-open completed ranges."""
    index = bisect_right(ranges, (ordinal, U64_MAX)) - 1
    return index >= 0 and ordinal < ranges[index][1]


def _sorted_uint_tuple_contains(values, target):
    """Find an exact uint in the sorted, duplicate-free tuple from slot parsing.

    _parse_slot_rows has already rejected non-uint, unsorted, and duplicate
    members. Keep the representation unchanged for snapshots and use a local
    lower-bound search for repeated data_ready membership queries.
    """
    count = len(values)
    if count == 0:
        return False
    if count == 1:
        return values[0] == target
    index = bisect_left(values, target)
    return index < count and values[index] == target


def _copy_consumed_slot_members(values):
    """Copy accepted immutable members once when a target owner is first touched."""
    return set(values)


def _freeze_consumed_slot_members(values):
    """Sort/freeze one touched target owner's final consumed members."""
    return tuple(sorted(values))


def _initial_cursor_run(descriptor, segment, run_index):
    """Compute prepare_batch's descriptor-only leading inactive prefix."""
    run = descriptor.endpoint_runs[run_index]
    shard_count, leg_count = len(run.shard_ordinals), len(run.legs)
    total = _run_endpoint_total(run)
    prefix = 0
    any_active = False
    for shard_index, shard_ordinal in enumerate(run.shard_ordinals):
        _selected_global_ordinal, shard = descriptor.selected_shard_tile(
            run.group_ordinal, run.first_tile - 1, shard_ordinal)
        for leg_index, leg in enumerate(run.legs):
            active = (leg.endpoint == "body_marker"
                      or (shard.emit_endpoint and shard.data_bytes != 0))
            if active:
                any_active = True
            if prefix == shard_index * leg_count + leg_index and not active:
                prefix += 1
    if not any_active:
        # The executor's prepare_batch check sees an all-inactive first tile
        # and retires the remaining monotone zero-work suffix compactly.
        prefix = total
    position = prefix
    if position == total:
        tile_offset, shard_index, leg_index, next_store = run.tile_count, 0, 0, None
    else:
        per_tile = shard_count * leg_count
        tile_offset, within = divmod(position, per_tile)
        shard_index, leg_index = divmod(within, leg_count)
        tile_ordinal = run.first_tile + tile_offset
        next_store = descriptor.endpoint_id(
            run_index, tile_ordinal, run.shard_ordinals[shard_index],
            leg_index) + 1
    initial_parent = (None if run.initial_parent_json_id is None
                      else run.initial_parent_json_id + 1)
    return {
        "rank": run.rank,
        "group_ordinal": run.group_ordinal,
        "first_tile": run.first_tile,
        "tile_count": run.tile_count,
        "completed_endpoint_count": prefix,
        "cancelled_endpoint_count": 0,
        "remaining_endpoint_count": total - prefix,
        "in_flight": False,
        "next_store_id": next_store,
        "tile_offset": tile_offset,
        "shard_index": shard_index,
        "leg_index": leg_index,
        "initial_parent_store_id": initial_parent,
        "completed_out_of_order": (),
        "active_endpoints": (),
    }


def _expected_endpoint_amount(descriptor, run_index, ordinal,
                              selected_shard_facts=None):
    run = descriptor.endpoint_runs[run_index]
    per_tile = len(run.shard_ordinals) * len(run.legs)
    tile_offset, within = divmod(ordinal, per_tile)
    shard_index, leg_index = divmod(within, len(run.legs))
    if selected_shard_facts is None:
        selected_shard_facts = descriptor.selected_shard_tile(
            run.group_ordinal, run.first_tile + tile_offset - 1,
            run.shard_ordinals[shard_index])
    shard = selected_shard_facts[1]
    leg = run.legs[leg_index].endpoint
    if leg == "body_marker":
        return {"ops": 1, "bytes": 1, "latency_ns": None}
    if leg in ("fill_ack_send", "fill_ack_recv"):
        amount = descriptor.protocol.fill_ack_bytes
    elif leg in ("release_ack_send", "release_ack_recv"):
        amount = descriptor.protocol.release_ack_bytes
    else:
        amount = shard.data_bytes
    return {"ops": None, "bytes": amount, "latency_ns": None}


def _parse_slot_rows(value: Any, path: str, descriptors: Mapping[str, Any],
                     selector_indexes: Optional[Mapping[str, Any]] = None) -> tuple:
    rows = _array(value, path)
    parsed = []
    for index, raw in enumerate(rows):
        p = f"{path}[{index}]"
        row = _mapping(raw, p, _SLOT_KEYS)
        domain = _string(row["buffer_domain_id"], p + ".buffer_domain_id")
        role = row["role"]
        if role not in ("source", "target"):
            raise PipelineLedgerError(f"{p}.role must be source or target")
        rank = _rank(row["rank"], p + ".rank")
        shard = _nullable_uint(row["shard_ordinal"], p + ".shard_ordinal")
        if (role == "source") != (shard is not None):
            raise PipelineLedgerError(f"{p}.shard_ordinal nullability disagrees with role")
        slot_index = _uint(row["slot_index"], p + ".slot_index", 1)
        occupied = _boolean(row["occupied"], p + ".occupied")
        consumed = _boolean(row["consumed"], p + ".consumed")
        targets = tuple(_uint(item, p + ".target_shards_to_consume[]")
                        for item in _array(
                            row["target_shards_to_consume"],
                            p + ".target_shards_to_consume"))
        consumed_targets = tuple(_uint(item, p + ".consumed_target_shards[]")
                                 for item in _array(
                                     row["consumed_target_shards"],
                                     p + ".consumed_target_shards"))
        if (any(left >= right for left, right in zip(targets, targets[1:]))
                or any(left >= right for left, right in zip(
                    consumed_targets, consumed_targets[1:]))):
            raise PipelineLedgerError(
                f"{p} target shard arrays must be sorted unique")
        if not set(consumed_targets).issubset(targets):
            raise PipelineLedgerError(
                f"{p}.consumed_target_shards is not a subset")
        owner = row["owner"]
        if occupied != (owner is not None):
            raise PipelineLedgerError(f"{p}.owner must be nonnull iff occupied")
        if owner is not None:
            owner = _mapping(owner, p + ".owner", _OWNER_KEYS)
            segment = _string(owner["segment_id"], p + ".owner.segment_id")
            descriptor = descriptors.get(segment)
            if descriptor is None:
                raise PipelineLedgerError(f"{p}.owner references unknown segment {segment!r}")
            ident = descriptor.identity
            pipeline_generation = _uint(
                owner["pipeline_generation"], p + ".owner.pipeline_generation")
            stage_epoch = _uint(owner["stage_epoch"], p + ".owner.stage_epoch")
            allocation = _uint(owner["staging_allocation_id"],
                               p + ".owner.staging_allocation_id")
            if (domain != ident.buffer_domain_id
                    or pipeline_generation != ident.generation
                    or stage_epoch != ident.stage_epoch
                    or allocation != ident.staging_allocation_id):
                raise PipelineLedgerError(f"{p}.owner disagrees with immutable descriptor identity")
            group_ordinal = _uint(owner["group_ordinal"], p + ".owner.group_ordinal")
            tile_ordinal = _uint(owner["tile_ordinal"], p + ".owner.tile_ordinal")
            selector_index = ((selector_indexes or {}).get(segment)
                              or _build_descriptor_selector_index(descriptor))
            group = selector_index.groups_by_ordinal.get(group_ordinal)
            if group is None or tile_ordinal < 1 or tile_ordinal > group.tile_axis.tile_count:
                raise PipelineLedgerError(f"{p}.owner points outside descriptor group/tile range")
            global_tile_ordinal = (
                group.tile_axis.global_tile_first + tile_ordinal - 1)
            if slot_index != (global_tile_ordinal - 1) % 2:
                raise PipelineLedgerError(
                    f"{p}.slot_index disagrees with two-slot tile parity")
            matching_run_index = _descriptor_run_at(
                selector_index, group_ordinal, rank, tile_ordinal)
            if role == "source":
                if (matching_run_index is None
                        or rank != descriptor.endpoint_runs[
                            matching_run_index].rank
                        or shard not in descriptor.endpoint_runs[
                            matching_run_index].shard_ordinals):
                    raise PipelineLedgerError(
                        f"{p} source shard is absent from the owner's endpoint run")
            else:
                if rank != ident.rank:
                    raise PipelineLedgerError(
                        f"{p} target slot rank disagrees with descriptor target rank")
                for target_shard_ordinal in targets:
                    if target_shard_ordinal >= len(group.shards):
                        raise PipelineLedgerError(
                            f"{p} target consumption shards disagree with descriptor tile")
                    _selected_global_ordinal, target_shard_tile = (
                        descriptor.selected_shard_tile(
                            group_ordinal, tile_ordinal - 1,
                            target_shard_ordinal))
                    if (not target_shard_tile.emit_endpoint
                            or target_shard_tile.target_rank != ident.rank):
                        raise PipelineLedgerError(
                            f"{p} target consumption shards disagree with descriptor tile")
        if consumed != bool(consumed_targets):
            raise PipelineLedgerError(
                f"{p}.consumed must match consumed target shards")
        if not occupied and (consumed or targets or consumed_targets):
            raise PipelineLedgerError(
                f"{p} empty slot cannot retain consume state")
        if role == "source" and (targets or consumed_targets or consumed):
            raise PipelineLedgerError(
                f"{p} source slot cannot carry target consume state")
        if occupied and role == "target" and not targets:
            raise PipelineLedgerError(
                f"{p} occupied target slot needs consumption authority")
        parsed_row = dict(row)
        parsed_row["owner"] = None if owner is None else dict(owner)
        parsed_row["target_shards_to_consume"] = targets
        parsed_row["consumed_target_shards"] = consumed_targets
        parsed.append(parsed_row)
    return _canonical_rows(parsed, lambda r: (r["buffer_domain_id"], r["role"], r["rank"], -1 if r["shard_ordinal"] is None else r["shard_ordinal"], r["slot_index"]), path)


def _parse_credit_rows(value: Any, path: str) -> tuple:
    rows = _array(value, path)
    parsed = []
    for i, raw in enumerate(rows):
        p = f"{path}[{i}]"
        row = _mapping(raw, p, _CREDIT_KEYS)
        capacity = _uint(row["capacity"], p + ".capacity")
        available = _uint(row["available"], p + ".available")
        if capacity != 2 or available > capacity:
            raise PipelineLedgerError(f"{p} requires capacity=2 and available in [0,2]")
        parsed_row = dict(row)
        parsed_row["buffer_domain_id"] = _string(
            row["buffer_domain_id"], p + ".buffer_domain_id")
        parsed_row["source_rank"] = _rank(
            row["source_rank"], p + ".source_rank")
        parsed_row["shard_ordinal"] = _uint(
            row["shard_ordinal"], p + ".shard_ordinal")
        parsed_row["outstanding_ack_count"] = _uint(
            row["outstanding_ack_count"], p + ".outstanding_ack_count")
        parsed.append(parsed_row)
    return _canonical_rows(parsed, lambda r: (r["buffer_domain_id"], r["source_rank"], r["shard_ordinal"]), path)


def _parse_obligations(value: Any, path: str, descriptors: Mapping[str, Any]) -> tuple:
    rows = _array(value, path)
    parsed = []
    for i, raw in enumerate(rows):
        p = f"{path}[{i}]"
        row = _mapping(raw, p, _OBLIGATION_KEYS)
        segment = _string(row["segment_id"], p + ".segment_id")
        if segment not in descriptors:
            raise PipelineLedgerError(f"{p} references unknown segment")
        count = _uint(row["remaining_endpoint_count"], p + ".remaining_endpoint_count")
        if count == 0 or row["state"] not in ("pending", "active"):
            raise PipelineLedgerError(f"{p} must be pending/active with positive remaining count")
        parsed_row = dict(row)
        parsed_row["obligation_id"] = _string(
            row["obligation_id"], p + ".obligation_id")
        parsed_row["rank"] = _rank(row["rank"], p + ".rank")
        parsed_row["first_store_id"] = _uint(
            row["first_store_id"], p + ".first_store_id")
        parsed.append(parsed_row)
    return _canonical_rows(parsed, lambda r: (r["segment_id"], r["rank"], r["first_store_id"], r["obligation_id"]), path)




def _parse_cursor_patch_run(raw_run: Any, path: str,
                            descriptor: PipelineDescriptor, segment: str,
                            run_index: int) -> dict:
    """Validate one sparse cursor run against its immutable descriptor."""
    run = _mapping(raw_run, path, _CURSOR_RUN_KEYS)
    expected = descriptor.endpoint_runs[run_index]
    rank = _rank(run["rank"], path + ".rank")
    group = _uint(run["group_ordinal"], path + ".group_ordinal")
    first_tile = _uint(run["first_tile"], path + ".first_tile")
    tile_count = _uint(run["tile_count"], path + ".tile_count")
    if (tile_count == 0 or rank != expected.rank
            or group != expected.group_ordinal or first_tile != expected.first_tile
            or tile_count != expected.tile_count):
        raise PipelineLedgerError(
            f"{path} does not match descriptor endpoint-run order/extent")
    completed = _uint(run["completed_endpoint_count"],
                      path + ".completed_endpoint_count")
    cancelled = _uint(run["cancelled_endpoint_count"],
                      path + ".cancelled_endpoint_count")
    remaining = _uint(run["remaining_endpoint_count"],
                      path + ".remaining_endpoint_count")
    in_flight = _boolean(run["in_flight"], path + ".in_flight")
    next_store = _nullable_uint(run["next_store_id"], path + ".next_store_id")
    initial_parent = _nullable_uint(
        run["initial_parent_store_id"], path + ".initial_parent_store_id")
    tile_offset = _uint(run["tile_offset"], path + ".tile_offset")
    shard_index = _uint(run["shard_index"], path + ".shard_index")
    leg_index = _uint(run["leg_index"], path + ".leg_index")
    shard_count, leg_count = len(expected.shard_ordinals), len(expected.legs)
    total_count = tile_count * shard_count * leg_count
    if total_count == 0 or tile_offset > tile_count:
        raise PipelineLedgerError(f"{path} has an invalid endpoint cursor extent")
    if tile_offset == tile_count:
        if shard_index != 0 or leg_index != 0 or next_store is not None:
            raise PipelineLedgerError(
                f"{path} exhausted cursor position must have null next store")
    elif shard_index >= shard_count or leg_index >= leg_count:
        raise PipelineLedgerError(f"{path} cursor position exceeds run axes")
    expected_parent = (None if expected.initial_parent_json_id is None
                       else expected.initial_parent_json_id + 1)
    if initial_parent != expected_parent:
        raise PipelineLedgerError(
            f"{path}.initial_parent_store_id disagrees with descriptor")
    ooo = _array(run["completed_out_of_order"],
                 path + ".completed_out_of_order")
    completed_ranges = [(0, completed)] if completed else []
    out_of_order_count = 0
    last_end = None
    for index, interval in enumerate(ooo):
        p = f"{path}.completed_out_of_order[{index}]"
        item = _mapping(interval, p, _OOO_KEYS)
        begin = _uint(item["first_ordinal"], p + ".first_ordinal")
        end = _uint(item["end_ordinal_exclusive"], p + ".end_ordinal_exclusive")
        if (begin >= end or begin < completed or end > total_count
                or (last_end is not None and begin < last_end)):
            raise PipelineLedgerError(f"{p} is empty, overlaps, or precedes completed prefix")
        last_end = end
        out_of_order_count += end - begin
        completed_ranges.append((begin, end))
    if completed + out_of_order_count + cancelled + remaining != total_count:
        raise PipelineLedgerError(
            f"{path} completed/out-of-order/remaining counts do not cover its endpoint run")
    endpoints = _array(run["active_endpoints"], path + ".active_endpoints")
    parsed_endpoints = []
    seen_store_ids = set()
    seen_transfer_ids = set()
    for index, active in enumerate(endpoints):
        p = f"{path}.active_endpoints[{index}]"
        item = _mapping(active, p, _ACTIVE_ENDPOINT_KEYS)
        store_id = _uint(item["store_id"], p + ".store_id")
        if store_id in seen_store_ids:
            raise PipelineLedgerError(f"{path} duplicates active store_id")
        seen_store_ids.add(store_id)
        active_tile = _uint(item["tile_ordinal"], p + ".tile_ordinal")
        active_shard = _uint(item["shard_ordinal"], p + ".shard_ordinal")
        endpoint = _string(item["endpoint"], p + ".endpoint")
        transfer_id = _uint(item["transfer_id"], p + ".transfer_id")
        if transfer_id in seen_transfer_ids:
            raise PipelineLedgerError(
                f"{path} duplicates active transfer_id")
        seen_transfer_ids.add(transfer_id)
        flow_id = _nullable_uint(item["flow_id"], p + ".flow_id")
        network_done = _boolean(item["network_done"], p + ".network_done")
        hbm_done = _boolean(item["hbm_done"], p + ".hbm_done")
        local_tile = active_tile - expected.first_tile
        try:
            shard_position = expected.shard_ordinals.index(active_shard)
        except ValueError as exc:
            raise PipelineLedgerError(
                f"{p}.shard_ordinal is absent from its endpoint run") from exc
        leg_position = next((i for i, leg in enumerate(expected.legs)
                             if leg.endpoint == endpoint), None)
        if (local_tile < 0 or local_tile >= expected.tile_count
                or leg_position is None):
            raise PipelineLedgerError(f"{p} endpoint position is outside its run")
        ordinal = (local_tile * shard_count * leg_count
                   + shard_position * leg_count + leg_position)
        if _completed_range_contains_half_open_sorted(
                completed_ranges, ordinal):
            raise PipelineLedgerError(f"{p} is simultaneously complete and active")
        expected_id = descriptor.endpoint_id(
            run_index, active_tile, active_shard, leg_position)
        if store_id != expected_id + 1:
            raise PipelineLedgerError(f"{p}.store_id disagrees with endpoint ID + 1")
        owner = _mapping(item["owner"], p + ".owner", _OWNER_KEYS)
        ident = descriptor.identity
        if (owner["segment_id"] != segment
                or owner["pipeline_generation"] != ident.generation
                or owner["stage_epoch"] != ident.stage_epoch
                or owner["staging_allocation_id"] != ident.staging_allocation_id
                or owner["group_ordinal"] != expected.group_ordinal
                or owner["tile_ordinal"] != active_tile):
            raise PipelineLedgerError(f"{p}.owner disagrees with its descriptor endpoint")
        if expected.legs[leg_position].endpoint != endpoint:
            raise PipelineLedgerError(f"{p}.endpoint disagrees with run leg")
        if endpoint != "body_marker":
            _selected_global_ordinal, shard_tile = (
                descriptor.selected_shard_tile(
                    expected.group_ordinal, active_tile - 1, active_shard))
            if not shard_tile.emit_endpoint:
                raise PipelineLedgerError(
                    f"{p} maps to an inactive stream-bytes endpoint")
        parsed_endpoint = dict(item)
        parsed_endpoint["flow_id"] = flow_id
        parsed_endpoint["network_done"] = network_done
        parsed_endpoint["hbm_done"] = hbm_done
        parsed_endpoint["owner"] = dict(owner)
        parsed_endpoints.append(parsed_endpoint)
    if in_flight != bool(endpoints):
        raise PipelineLedgerError(f"{path}.in_flight must match active_endpoints")
    if tile_offset < tile_count:
        tile_ordinal = expected.first_tile + tile_offset
        shard = expected.shard_ordinals[shard_index]
        leg = expected.legs[leg_index]
        if leg.endpoint != "body_marker":
            _selected_global_ordinal, shard_tile = (
                descriptor.selected_shard_tile(
                    expected.group_ordinal, tile_ordinal - 1, shard))
            if not shard_tile.emit_endpoint:
                raise PipelineLedgerError(
                    f"{path} cursor position points to an inactive endpoint")
        expected_next = descriptor.endpoint_id(
            run_index, tile_ordinal, shard, leg_index) + 1
        if next_store != expected_next:
            raise PipelineLedgerError(
                f"{path}.next_store_id disagrees with cursor position")
    parsed_run = dict(run)
    parsed_run["completed_out_of_order"] = tuple(dict(x) for x in ooo)
    parsed_run["active_endpoints"] = tuple(parsed_endpoints)
    return parsed_run


def _parse_cursor_patches(value: Any, path: str,
                          descriptors: Mapping[str, Any]) -> tuple:
    rows = _array(value, path)
    parsed = []
    for index, raw in enumerate(rows):
        p = f"{path}[{index}]"
        patch = _mapping(raw, p, frozenset({
            "segment_id", "cursor_version", "retired", "runs"}))
        segment = _string(patch["segment_id"], p + ".segment_id")
        descriptor = descriptors.get(segment)
        if descriptor is None:
            raise PipelineLedgerError(f"{p} references unknown descriptor segment")
        version = _uint(patch["cursor_version"], p + ".cursor_version")
        retired = _boolean(patch["retired"], p + ".retired")
        run_rows = _array(patch["runs"], p + ".runs")
        parsed_runs = []
        previous_index = None
        for run_index, raw_run in enumerate(run_rows):
            q = f"{p}.runs[{run_index}]"
            item = _mapping(raw_run, q, frozenset({"run_index", "value"}))
            ordinal = _uint(item["run_index"], q + ".run_index")
            if ordinal >= len(descriptor.endpoint_runs):
                raise PipelineLedgerError(f"{q}.run_index exceeds descriptor runs")
            raw_value = item["value"]
            parsed_value = (None if raw_value is None else
                            _parse_cursor_patch_run(
                                raw_value, q + ".value", descriptor, segment,
                                ordinal))
            if parsed_value is None and not retired:
                raise PipelineLedgerError(
                    f"{q}.value may be null only during descriptor retirement")
            if parsed_value is not None and retired:
                raise PipelineLedgerError(
                    f"{q}.value must be null on descriptor retirement")
            if previous_index is not None and ordinal <= previous_index:
                raise PipelineLedgerError(
                    f"{p}.runs must be sorted and duplicate-free")
            previous_index = ordinal
            parsed_runs.append((ordinal, parsed_value))
        if retired and (len(parsed_runs) != len(descriptor.endpoint_runs)
                        or (parsed_runs and parsed_runs[0][0] != 0)
                        or (parsed_runs and parsed_runs[-1][0]
                            != len(descriptor.endpoint_runs) - 1)
                        or any(run is not None for _, run in parsed_runs)):
            raise PipelineLedgerError(
                f"{p} retirement must tombstone every descriptor cursor run")
        if parsed and segment <= parsed[-1]["segment_id"]:
            raise PipelineLedgerError(
                f"{path} must be sorted and duplicate-free")
        parsed.append({"segment_id": segment, "cursor_version": version,
                       "retired": retired, "runs": tuple(parsed_runs)})
    return tuple(parsed)


def _resolve_selector(descriptor: PipelineDescriptor,
                      selector: Mapping[str, Any], indexes=None,
                      rank=None) -> tuple[int, int, int]:
    """Resolve a selector to (old JSON ID, run ordinal, leg ordinal)."""
    group = _uint(selector["group_ordinal"], "endpoint_selector.group_ordinal")
    tile = _uint(selector["tile_ordinal"], "endpoint_selector.tile_ordinal")
    shard = _uint(selector["shard_ordinal"], "endpoint_selector.shard_ordinal")
    endpoint = _string(selector["endpoint"], "endpoint_selector.endpoint")
    indexes = indexes or _build_descriptor_selector_index(descriptor)
    if rank is not None:
        match = _descriptor_selector_at(
            indexes, group, rank, tile, shard, endpoint)
    else:
        matches = []
        # Used only by small direct parser helpers. Production parsing passes
        # rank from its already typed transition/resource row.
        for key in indexes.selector_intervals:
            if key[0] == group and key[2] == shard and key[3] == endpoint:
                found = _descriptor_selector_at(
                    indexes, group, key[1], tile, shard, endpoint)
                if found is not None:
                    matches.append(found)
        match = matches[0] if len(matches) == 1 else None
    if match is None:
        raise PipelineLedgerError("endpoint selector must map to exactly one descriptor endpoint")
    run_ordinal, leg_ordinal = match
    return (descriptor.endpoint_id(run_ordinal, tile, shard, leg_ordinal),
            run_ordinal, leg_ordinal)


def _parse_one_pipeline_run(raw, p, section_path, descriptors,
                            selector_indexes, *, protocol_version,
                            previous_seq) -> dict:
    """Validate one wire pipeline run into the facts prepare consumes.

    ``p`` is the row path for field errors; ``section_path`` names the array
    for the cross-row sequence-order error.  ``previous_seq`` is the previous
    run's ``first_logical_seq`` (None for the first row).
    """
    row = _mapping(raw, p, _RUN_KEYS)
    seq = _uint(row["first_logical_seq"], p + ".first_logical_seq")
    if seq == 0:
        raise PipelineLedgerError(f"{p}.first_logical_seq must start at 1")
    _string(row["pattern_id"], p + ".pattern_id")
    repeat = _uint(row["repeat_count"], p + ".repeat_count")
    if repeat != 1:
        raise PipelineLedgerError(f"{p}.repeat_count must equal 1")
    transition = row["transition_kind"]
    if transition not in _TRANSITIONS:
        raise PipelineLedgerError(f"{p}.transition_kind is unknown")
    segment = _string(row["descriptor_segment_id"], p + ".descriptor_segment_id")
    descriptor = descriptors.get(segment)
    if descriptor is None:
        raise PipelineLedgerError(f"{p} references undeclared descriptor {segment!r}")
    selector = row["endpoint_selector"]
    store = row["store_id_first"]
    store_stride = row["store_id_stride"]
    legacy = row["legacy_context"]
    selector_match = None
    endpoint_ordinal = None
    selected_global_ordinal = None
    selected_shard_facts = None
    endpoint_store_id = None
    endpoint_fields = (selector, store, store_stride, legacy)
    if all(item is None for item in endpoint_fields):
        for field in ("tile_first", "tile_stride", "rank_first", "rank_stride", "shard_first", "shard_stride"):
            if row[field] is not None:
                raise PipelineLedgerError(f"{p}.{field} must be null on descriptor transition")
        if row["resource_leg"] is not None:
            raise PipelineLedgerError(f"{p}.resource_leg must be null on descriptor transition")
    elif any(item is None for item in endpoint_fields):
        raise PipelineLedgerError(f"{p} endpoint selector/store/context fields must be all null or all present")
    else:
        selector = _mapping(selector, p + ".endpoint_selector", frozenset({"group_ordinal", "tile_ordinal", "shard_ordinal", "endpoint"}))
        selector_index = ((selector_indexes or {}).get(segment)
                          or _build_descriptor_selector_index(descriptor))
        rank_value = _rank(row["rank_first"], p + ".rank_first")
        json_id, run_ordinal, leg_ordinal = _resolve_selector(
            descriptor, selector, selector_index, rank_value)
        selector_match = (run_ordinal, leg_ordinal)
        tile = _uint(row["tile_first"], p + ".tile_first")
        rank = rank_value
        shard = _uint(row["shard_first"], p + ".shard_first")
        if (tile != selector["tile_ordinal"] or shard != selector["shard_ordinal"]
                or rank != descriptor.endpoint_runs[run_ordinal].rank):
            raise PipelineLedgerError(f"{p} affine selector axes disagree")
        for field in ("tile_stride", "rank_stride", "shard_stride", "store_id_stride"):
            if _uint(row[field], p + "." + field) != 0:
                raise PipelineLedgerError(f"{p}.{field} must be 0 while repeat_count=1")
        endpoint_store_id = _uint(store, p + ".store_id_first")
        if endpoint_store_id != json_id + 1:
            raise PipelineLedgerError(f"{p}.store_id_first must equal endpoint JSON ID + 1")
        legacy = _mapping(legacy, p + ".legacy_context", frozenset({"request_id", "stage", "generation"}))
        endpoint_context = descriptor.endpoint_runs[run_ordinal].legacy_context
        if (legacy["request_id"], legacy["stage"], legacy["generation"]) != (endpoint_context.request_id, endpoint_context.stage, endpoint_context.generation):
            raise PipelineLedgerError(f"{p}.legacy_context disagrees with endpoint run")
        endpoint_run = descriptor.endpoint_runs[run_ordinal]
        shard_position = endpoint_run.shard_ordinals.index(
            selector["shard_ordinal"])
        endpoint_ordinal = (
            (selector["tile_ordinal"] - endpoint_run.first_tile)
            * len(endpoint_run.shard_ordinals) * len(endpoint_run.legs)
            + shard_position * len(endpoint_run.legs)
            + selector_match[1])
    if row["resource_leg"] not in _RESOURCE_LEGS:
        raise PipelineLedgerError(f"{p}.resource_leg is unknown")
    tick = _uint(row["tick_first"], p + ".tick_first")
    if _uint(row["tick_stride"], p + ".tick_stride") != 0:
        raise PipelineLedgerError(f"{p}.tick_stride must be 0")
    cv0 = _uint(row["cursor_version_start"], p + ".cursor_version_start")
    cv1 = _uint(row["cursor_version_end"], p + ".cursor_version_end")
    if cv1 < cv0 or cv1 > cv0 + 1:
        raise PipelineLedgerError(f"{p} cursor version transition must be stable or increment by one")
    prefix = _array(row["resource_quantity_prefix"], p + ".resource_quantity_prefix")
    if len(prefix) != 2:
        raise PipelineLedgerError(f"{p}.resource_quantity_prefix length must equal repeat_count+1")
    baseline = _amount(prefix[0], p + ".resource_quantity_prefix[0]")
    quantity = _amount(prefix[1], p + ".resource_quantity_prefix[1]")
    for dim in ("ops", "bytes", "latency_ns"):
        if (baseline[dim] is None) != (quantity[dim] is None):
            raise PipelineLedgerError(f"{p} resource prefix changes {dim} nullability")
        if quantity[dim] is None:
            if baseline[dim] is not None:
                raise PipelineLedgerError(f"{p} undefined {dim} baseline must be null")
        elif type(baseline[dim]) is not int or baseline[dim] != 0:
            raise PipelineLedgerError(f"{p} defined {dim} baseline must be integer zero")
    if row["resource_leg"] is None and any(
            quantity[dim] is not None
            for dim in ("ops", "bytes", "latency_ns")):
        raise PipelineLedgerError(
            f"{p} resource quantities require a resource_leg")
    if row["resource_leg"] is not None and endpoint_ordinal is not None:
        # Keep only immutable descriptor-derived facts for prepare; the
        # mutable slot/credit/ready state remains checked at application.
        selected_global_ordinal, selected_shard_facts = (
            descriptor.selected_shard_tile(
                selector["group_ordinal"],
                selector["tile_ordinal"] - 1,
                selector["shard_ordinal"]))
        expected_amount = _expected_endpoint_amount(
            descriptor, run_ordinal, endpoint_ordinal,
            (selected_global_ordinal, selected_shard_facts))
        if not _typed_equal(quantity, expected_amount):
            raise PipelineLedgerError(
                f"{p} resource quantity does not match descriptor endpoint")
        endpoint_name = descriptor.endpoint_runs[run_ordinal].legs[
            selector_match[1]].endpoint
        if ((row["resource_leg"] == "body_marker")
                != (endpoint_name == "body_marker")):
            raise PipelineLedgerError(
                f"{p}.resource_leg disagrees with descriptor endpoint kind")
    if protocol_version == 4:
        slot_patches = _parse_v4_slot_patches(
            row["slot_patches"], p + ".slot_patches",
            descriptors, selector_indexes)
    else:
        slot_patches = _parse_row_patches(
            row["slot_patches"], p + ".slot_patches",
            lambda values, q: _parse_slot_rows(
                values, q, descriptors, selector_indexes),
            lambda value: (value["buffer_domain_id"], value["role"],
                           value["rank"], value["shard_ordinal"],
                           value["slot_index"]),
            lambda key: (key[0], key[1], key[2],
                         -1 if key[3] is None else key[3], key[4]))
    credit_patches = _parse_row_patches(
        row["credit_patches"], p + ".credit_patches",
        _parse_credit_rows,
        lambda value: (value["buffer_domain_id"], value["source_rank"],
                       value["shard_ordinal"]), lambda key: key)
    obligation_patches = _parse_row_patches(
        row["obligation_patches"], p + ".obligation_patches",
        lambda values, q: _parse_obligations(values, q, descriptors),
        lambda value: value["obligation_id"], lambda key: key)
    # Schema-level transition invariants; cross-delivery pre-state checks
    # are performed by apply_envelope against the accepted snapshot.
    descriptor_level = transition in (
        "descriptor_commit", "service_done", "request_close", "settle_done",
        "descriptor_retire",
    ) or (transition == "consume_done" and selector is None)
    if descriptor_level and (selector is not None or row["resource_leg"] is not None):
        raise PipelineLedgerError(f"{p} descriptor-level transition cannot bind an endpoint or resource")
    if transition == "endpoint_done" and selector is None:
        raise PipelineLedgerError(f"{p} endpoint_done transition requires a concrete selector")
    if transition not in (
            "descriptor_commit", "service_done", "request_close", "settle_done",
            "descriptor_retire") and selector is None and transition != "consume_done":
        raise PipelineLedgerError(f"{p} endpoint transition requires a concrete selector")
    if selector is not None and selected_shard_facts is None:
        # Selector-only transitions still need immutable geometry during
        # prepare; defer this until their complete wire row has validated.
        selected_global_ordinal, selected_shard_facts = (
            descriptor.selected_shard_tile(
                selector["group_ordinal"],
                selector["tile_ordinal"] - 1,
                selector["shard_ordinal"]))
    # Keep only the validated facts consumed by prepare_envelope. The
    # original wire row remains available to _parse_envelope for exact
    # canonical replay bytes, and all raw fields were checked above.
    parsed_row = {
        "first_logical_seq": seq,
        "descriptor_segment_id": segment,
        "transition_kind": transition,
        "endpoint_selector": (
            None if selector is None else dict(selector)),
        "endpoint_run_index": (
            None if selector_match is None else selector_match[0]),
        "endpoint_leg_index": (
            None if selector_match is None else selector_match[1]),
        "endpoint_ordinal": endpoint_ordinal,
        "_selected_global_tile_ordinal": selected_global_ordinal,
        "_selected_shard_tile": selected_shard_facts,
        "endpoint_store_id": endpoint_store_id,
        "slot_patches": slot_patches,
        "credit_patches": credit_patches,
        "obligation_patches": obligation_patches,
        "tick_first": tick,
        "cursor_version_start": cv0,
        "cursor_version_end": cv1,
    }
    if previous_seq is not None and seq <= previous_seq:
        raise PipelineLedgerError(
            f"{section_path} must preserve logical sequence order")
    return parsed_row


def _parse_pipeline_runs(value: Any, descriptors: Mapping[str, Any], path: str,
                         selector_indexes: Optional[Mapping[str, Any]] = None,
                         *, protocol_version: int = 3) -> tuple:
    rows = _array(value, path)
    parsed = []
    previous_seq = None
    for i, raw in enumerate(rows):
        parsed.append(_parse_one_pipeline_run(
            raw, f"{path}[{i}]", path, descriptors, selector_indexes,
            protocol_version=protocol_version, previous_seq=previous_seq))
        previous_seq = parsed[-1]["first_logical_seq"]
    return tuple(parsed)


def _parse_resources(value: Any, descriptors: Mapping[str, Any], path: str,
                     selector_indexes: Optional[Mapping[str, Any]] = None) -> tuple:
    rows = _array(value, path)
    parsed = []
    allowed_states = {
        "noc_flow": {"Active", "PropagatingTail", "Completed"},
        "hbm_job": {"issued", "in_service"},
        "pool_job": {"LatencyWaiting", "Streaming", "FluidCompleteAwaitingCallback"},
    }
    for i, raw in enumerate(rows):
        p = f"{path}[{i}]"
        if not issubclass(type(raw), _RuntimeMapping):
            raise PipelineLedgerError(f"{p} must be an object")
        kind = raw.get("resource_type")
        if kind not in _RESOURCE_EXTRAS:
            raise PipelineLedgerError(f"{p}.resource_type is unknown")
        row = _mapping(raw, p, _RESOURCE_COMMON | _RESOURCE_EXTRAS[kind])
        if row["state"] not in allowed_states[kind]:
            raise PipelineLedgerError(f"{p}.state is invalid for {kind}")
        for field in ("work_id", "request_id", "operation", "leg", "progress_source", "service_unit"):
            _string(row[field], p + "." + field)
        if not isinstance(row["stage"], str):
            raise PipelineLedgerError(f"{p}.stage must be a string")
        if row["service_unit"] != "bytes":
            raise PipelineLedgerError(f"{p}.service_unit must be bytes")
        _uint(row["generation"], p + ".generation")
        _rank(row["rank"], p + ".rank")
        _uint(row["node_id"], p + ".node_id")
        _uint(row["as_of_tick"], p + ".as_of_tick")
        _nullable_uint(row["issued_tick"], p + ".issued_tick")
        if row["progress_source"] != "authoritative":
            raise PipelineLedgerError(f"{p}.progress_source must be authoritative")
        _amount(row["total"], p + ".total")
        _amount(row["remaining"], p + ".remaining")
        route = _array(row["route"], p + ".route")
        for j, link in enumerate(route):
            _uint(link, f"{p}.route[{j}]")
        segment, selector = row["descriptor_segment_id"], row["endpoint_selector"]
        if (segment is None) != (selector is None):
            raise PipelineLedgerError(f"{p} descriptor segment and endpoint selector must be both null or present")
        if segment is not None:
            segment = _string(segment, p + ".descriptor_segment_id")
            descriptor = descriptors.get(segment)
            if descriptor is None:
                raise PipelineLedgerError(f"{p} references unknown descriptor")
            selector = _mapping(selector, p + ".endpoint_selector", frozenset({"group_ordinal", "tile_ordinal", "shard_ordinal", "endpoint"}))
            selector_index = ((selector_indexes or {}).get(segment)
                              or _build_descriptor_selector_index(descriptor))
            json_id, run_ordinal, _leg_ordinal = _resolve_selector(
                descriptor, selector, selector_index, row["rank"])
            run = descriptor.endpoint_runs[run_ordinal]
            context = run.legacy_context
            if row["node_id"] != json_id + 1 or row["rank"] != run.rank:
                raise PipelineLedgerError(f"{p} pipeline NodeStore mapping is inconsistent")
            if (row["request_id"], row["stage"], row["generation"]) != (context.request_id, context.stage, context.generation):
                raise PipelineLedgerError(f"{p} pipeline context disagrees with endpoint legacy_context")
            if kind == "noc_flow" and not row["leg"].startswith("noc:"):
                raise PipelineLedgerError(f"{p} NoC resource leg name is invalid")
            if kind == "hbm_job" and not row["leg"].startswith("hbm:"):
                raise PipelineLedgerError(f"{p} HBM resource leg name is invalid")
            if kind == "pool_job" and not row["leg"].startswith("pool:"):
                raise PipelineLedgerError(f"{p} pool resource leg name is invalid")
        else:
            selector = None
            is_unattributed_noc = (
                kind == "noc_flow" and row["request_id"] == "noc"
                and row["stage"] == "" and row["generation"] == 0)
            # §14.3 裁决的外生背景 hbm 腿（Agent R C++ 导出侧同步）：
            # 身份解析元数据失败且 transfer-sink COMM_READ/WRITE 的后端
            # job 按 issue_seq 真实身份字面导出——恰此一种空 stage hbm 行
            # 合法，其余空 stage hbm 行保持 fail-closed。
            is_external_background_hbm = (
                kind == "hbm_job"
                and row["request_id"] == "external_background"
                and row["stage"] == ""
                and row["generation"] == 0)
            if not (is_unattributed_noc or is_external_background_hbm) \
                    and row["stage"] not in (
                        "prefill", "decode", "completion"):
                raise PipelineLedgerError(f"{p}.stage is invalid")
        if segment is not None and row["stage"] not in (
                "prefill", "decode", "completion"):
            raise PipelineLedgerError(f"{p}.stage is invalid")
        if kind != "noc_flow" and route:
            raise PipelineLedgerError(f"{p} non-NoC route must be empty")
        expected_work_id = leg_work_id(
            canonical_work_id(row["request_id"], row["stage"],
                             row["generation"], row["rank"], row["node_id"]),
            row["leg"])
        if row["work_id"] != expected_work_id:
            raise PipelineLedgerError(f"{p}.work_id is not canonical for its typed identity/leg")
        if kind != "hbm_job":
            for amount_name in ("total", "remaining"):
                amount = row[amount_name]
                if amount["ops"] is not None or amount["latency_ns"] is not None:
                    raise PipelineLedgerError(f"{p}.{amount_name} for {kind} may define bytes only")
        if kind == "noc_flow":
            flow_id = _uint(row["flow_id"], p + ".flow_id")
            if row["leg"] != f"noc:{flow_id}":
                raise PipelineLedgerError(f"{p}.leg must identify its original flow_id")
            if segment is None and not (
                    row["request_id"] == "noc" and row["stage"] == ""
                    and row["generation"] == 0 and row["node_id"] == flow_id):
                raise PipelineLedgerError(
                    f"{p} unattributed NoC identity must preserve noc/empty-stage/flow-id form")
            _finite_nonnegative(row["current_rate_Bpns"], p + ".current_rate_Bpns")
            _finite_nonnegative(row["last_rate_update_time"], p + ".last_rate_update_time")
            if row["state"] == "Completed" and row["remaining"]["bytes"] != 0:
                raise PipelineLedgerError(f"{p} completed NoC tail must have zero remaining bytes")
        elif kind == "hbm_job":
            _string(row["kind"], p + ".kind")
            if row["request_id"] == "external_background" \
                    and row["stage"] == "" and row["generation"] == 0:
                # §14.3 外生背景腿的字面身份：rank 必须与行自身一致，
                # issue_seq 为正 uint（后端 job 序号）；kind 仍须是
                # transfer-sink 的 COMM_READ/COMM_WRITE。其余校验
                # （amounts/canonical work_id/start_tick）与普通 hbm 行
                # 同规，不放宽。
                if row["kind"] not in ("COMM_READ", "COMM_WRITE"):
                    raise PipelineLedgerError(
                        f"{p} external background HBM job kind must be a "
                        "transfer-sink COMM_READ/COMM_WRITE")
                leg_rank, _, leg_seq = row["leg"].partition("#")
                leg_rank = leg_rank.partition(":")
                if (not str(row["leg"]).startswith("hbm_external:")
                        or not leg_seq.isdigit() or int(leg_seq) < 1
                        or leg_rank[0] != "hbm_external"
                        or not leg_rank[2].isdigit()
                        or int(leg_rank[2]) != row["rank"]):
                    raise PipelineLedgerError(
                        f"{p}.leg must be the literal external background "
                        f"form hbm_external:{{rank}}#{{issue_seq}} with the "
                        "row's own rank and a positive backend issue_seq")
            elif row["leg"] != f"hbm:{row['kind']}":
                raise PipelineLedgerError(f"{p}.leg must identify its HBM job kind")
            _uint(row["start_tick"], p + ".start_tick")
        else:
            port = _uint(row["port_index"], p + ".port_index")
            issue_seq = _uint(row["issue_seq"], p + ".issue_seq")
            if row["leg"] != f"pool:{port}:{issue_seq}":
                raise PipelineLedgerError(f"{p}.leg must identify its original pool job")
            _finite_nonnegative(row["latency_ready_ns"], p + ".latency_ready_ns")
        for dimension in ("ops", "bytes", "latency_ns"):
            total = row["total"][dimension]
            remaining = row["remaining"][dimension]
            if (total is None) != (remaining is None):
                raise PipelineLedgerError(f"{p} changes {dimension} applicability")
            if total is not None and remaining > total:
                raise PipelineLedgerError(f"{p} remaining {dimension} exceeds total")
        parsed.append(dict(row))
    return _canonical_rows(parsed, lambda r: ("" if r["descriptor_segment_id"] is None else r["descriptor_segment_id"], r["rank"], r["node_id"], r["leg"], r["resource_type"]), path)


_EnvelopeHeader = namedtuple("_EnvelopeHeader", (
    "row", "canonical_bytes", "replay_token", "epoch", "delivery_sequence",
    "snapshot_kind", "tick", "base_watermark", "seq_start", "seq_end",
    "event_count", "base_version", "result_version"))


def _validate_envelope_header(envelope: Any, *, protocol_version: int
                              ) -> _EnvelopeHeader:
    """Validate every envelope-level scalar field and derive header facts.

    Shared by the exact offline decoder (``_parse_envelope``) and the fused
    streaming production path.  ``canonical_bytes`` is computed here so both
    consumers share one bounded replay representation; the per-row sections
    stay owned by their callers.
    """
    if protocol_version not in (3, 4):
        raise PipelineLedgerError("unsupported pipeline ledger decoder version")
    if not issubclass(type(envelope), _RuntimeMapping):
        raise PipelineLedgerError("pipeline ledger envelope must be an object")
    required = _TOP_KEYS - {"optional_sha256"}
    missing = required - set(envelope)
    unknown = set(envelope) - _TOP_KEYS
    if missing or unknown:
        raise PipelineLedgerError(
            "pipeline ledger envelope keys invalid "
            f"(missing={sorted(missing)}, unknown={sorted(unknown)})")
    row = envelope
    if protocol_version == 4:
        expected_identity = (4, 4, "bridge-v4-20261011",
                             "workledger-v4-20261011")
    else:
        expected_identity = (3, 3, "bridge-v3-20261009",
                             "workledger-v3-20261009")
    if (row["producer"] != PRODUCER
            or _uint(row["schema_version"], "schema_version")
                != expected_identity[0]
            or _uint(row["protocol_version"], "protocol_version")
                != expected_identity[1]):
        raise PipelineLedgerError(
            f"unsupported v{protocol_version} producer/schema/protocol")
    if (row["bridge_protocol"] != expected_identity[2]
            or row["work_ledger_schema"] != expected_identity[3]):
        raise PipelineLedgerError(
            f"unsupported v{protocol_version} bridge or work-ledger schema")
    if _uint(row["descriptor_version"], "descriptor_version") != DESCRIPTOR_VERSION:
        raise PipelineLedgerError("unsupported descriptor_version")
    epoch = _string(row["epoch"], "epoch")
    delivery = _uint(row["delivery_sequence"], "delivery_sequence")
    tick = _uint(row["tick"], "tick")
    kind = row["snapshot_kind"]
    if kind not in ("full", "delta"):
        raise PipelineLedgerError("snapshot_kind must be full or delta")
    watermark = _uint(row["base_watermark"], "base_watermark")
    seq_start = _uint(row["logical_seq_start"], "logical_seq_start")
    seq_end = _uint(row["logical_seq_end"], "logical_seq_end")
    count = _uint(row["logical_event_count"], "logical_event_count")
    base_version = _uint(row["base_state_version"], "base_state_version")
    result_version = _uint(row["result_state_version"], "result_state_version")
    replay = _string(row["replay_token"], "replay_token")
    if replay != f"{PRODUCER}/{epoch}/{delivery}":
        raise PipelineLedgerError("replay_token does not match producer/epoch/delivery")
    if (count == 0 and (seq_start != watermark + 1 or seq_end != watermark)):
        raise PipelineLedgerError("empty delta sequence interval is invalid")
    if (count > 0 and (seq_start != watermark + 1 or seq_end != seq_start + count - 1)):
        raise PipelineLedgerError("logical sequence interval/count is inconsistent")
    if (kind == "full" and watermark != 0) or result_version not in (base_version, base_version + 1):
        raise PipelineLedgerError("full watermark or state-version transition is invalid")
    if "optional_sha256" in row:
        digest = row["optional_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise PipelineLedgerError("optional_sha256 must be 64 lowercase hexadecimal characters")
        without_digest = {k: v for k, v in row.items() if k != "optional_sha256"}
        actual = hashlib.sha256(_canonical_envelope_bytes(without_digest)).hexdigest()
        if actual != digest:
            raise PipelineLedgerError("optional_sha256 does not match canonical envelope JSON")
    return _EnvelopeHeader(
        row=row,
        canonical_bytes=_canonical_envelope_bytes(row),
        replay_token=replay,
        epoch=epoch,
        delivery_sequence=delivery,
        snapshot_kind=kind,
        tick=tick,
        base_watermark=watermark,
        seq_start=seq_start,
        seq_end=seq_end,
        event_count=count,
        base_version=base_version,
        result_version=result_version)


def _parse_descriptor_commits(row, retained_descriptors,
                              retained_selector_indexes, *,
                              allow_retained_replay: bool):
    """Parse ``descriptor_commits`` into shared declaration facts.

    Returns ``(descriptors, selector_index_additions,
    dependency_index_additions, candidate_descriptors, selector_indexes)``
    where the two ChainMaps already merge this delivery's declarations with
    the retained registry for endpoint-bound row lookups.
    """
    descriptors = []
    seen_segments = set()
    for i, raw in enumerate(_array(row["descriptor_commits"], "descriptor_commits")):
        p = f"descriptor_commits[{i}]"
        commit = _mapping(raw, p, frozenset({"identity", "descriptor", "commit_tick", "cursor_version", "service_done", "settle_done", "retired"}))
        descriptor = PipelineDescriptor.from_wire(commit["descriptor"])
        ident = descriptor.identity
        identity = _mapping(commit["identity"], p + ".identity", frozenset({"request_id", "stage", "generation", "stage_epoch", "segment_ordinal", "rank", "staging_allocation_id", "segment_id", "buffer_domain_id"}))
        if dict(identity) != descriptor.to_wire()["identity"]:
            raise PipelineLedgerError(f"{p}.identity differs from immutable descriptor")
        if ident.segment_id in seen_segments:
            raise PipelineLedgerError(f"duplicate descriptor declaration {ident.segment_id}")
        seen_segments.add(ident.segment_id)
        _uint(commit["commit_tick"], p + ".commit_tick")
        _uint(commit["cursor_version"], p + ".cursor_version")
        _boolean(commit["service_done"], p + ".service_done")
        _boolean(commit["settle_done"], p + ".settle_done")
        _boolean(commit["retired"], p + ".retired")
        descriptors.append((descriptor, dict(commit)))
    declared_descriptors = {}
    for desc, _commit in descriptors:
        segment_id = desc.identity.segment_id
        prior = (retained_descriptors or {}).get(segment_id)
        if prior is not None and prior.to_wire() != desc.to_wire():
            raise PipelineLedgerError("immutable descriptor changed after declaration")
        if prior is not None and not allow_retained_replay:
            raise PipelineLedgerError("descriptor segment was declared more than once")
        declared_descriptors[segment_id] = desc
    candidate_descriptors = ChainMap(declared_descriptors, retained_descriptors or {})
    selector_index_additions = {}
    for segment_id, descriptor in declared_descriptors.items():
        retained_index = (retained_selector_indexes or {}).get(segment_id)
        if retained_index is None:
            selector_index_additions[segment_id] = (
                _build_descriptor_selector_index(descriptor))
    selector_indexes = ChainMap(
        selector_index_additions, retained_selector_indexes or {})
    dependency_index_additions = {
        segment_id: _build_descriptor_dependency_index(descriptor)
        for segment_id, descriptor in declared_descriptors.items()
        if segment_id not in (retained_descriptors or {})
    }
    return (descriptors, selector_index_additions,
            dependency_index_additions, candidate_descriptors,
            selector_indexes)


def _parse_envelope(envelope: Any, retained_descriptors=None,
                    retained_selector_indexes=None,
                    *, allow_retained_replay: bool = False,
                    protocol_version: int = 3) -> _ParsedEnvelope:
    """Exact offline decoder: materialize every node event and pipeline run."""
    header = _validate_envelope_header(
        envelope, protocol_version=protocol_version)
    row = header.row
    delivery = header.delivery_sequence
    tick = header.tick
    (descriptors, selector_index_additions, dependency_index_additions,
     candidate_descriptors, selector_indexes) = _parse_descriptor_commits(
        row, retained_descriptors, retained_selector_indexes,
        allow_retained_replay=allow_retained_replay)
    # Endpoint-bound transition/resource rows may refer to descriptors accepted
    # in previous deliveries. The live-state lookup is merged by PipelineLedger.
    node_rows = _array(row["node_transitions"], "node_transitions")
    node_events = []
    allowed_payload = EVENT_PAYLOAD_WHITELIST - {"kind", "work_id", "tick", "event_id"}
    for i, raw in enumerate(node_rows):
        p = f"node_transitions[{i}]"
        event = _mapping(raw, p, _NODE_ROW_KEYS)
        seq = _uint(event["logical_seq"], p + ".logical_seq")
        etick = _uint(event["tick"], p + ".tick")
        if seq == 0 or etick > tick:
            raise PipelineLedgerError(f"{p} logical sequence/tick is invalid")
        event_kind = _string(event["kind"], p + ".kind")
        if event_kind not in _NODE_EVENT_KINDS:
            raise PipelineLedgerError(f"{p}.kind is unknown")
        work_id = _string(event["work_id"], p + ".work_id")
        payload = event["payload"]
        if (not issubclass(type(payload), _RuntimeMapping)
                or not all(isinstance(key, str) for key in payload)):
            raise PipelineLedgerError(f"{p}.payload must be a string-keyed object")
        unknown = set(payload) - allowed_payload
        if unknown:
            raise PipelineLedgerError(f"{p}.payload has unknown keys {sorted(unknown)}")
        try:
            # The row, scalar headers and body key whitelist above establish
            # the v3 header contract before entering the private constructor.
            # WorkTxn retains only its final node event for adjacent legacy
            # replay. The full frame canonical bytes above still validate all
            # input fields before any transaction begins.
            typed = _typed_v3_node_event_from_validated_scalars(
                kind=event_kind,
                work_id=work_id,
                tick=etick,
                delivery_sequence=delivery,
                producer=PRODUCER,
                local_seq=seq,
                payload=payload,
                build_canonical_content=(i == len(node_rows) - 1))
        except (WorkLedgerError, TypeError, ValueError) as exc:
            raise PipelineLedgerError(f"{p} is not a valid typed node event: {exc}") from exc
        if node_events and seq <= node_events[-1][0]:
            raise PipelineLedgerError(
                "node_transitions must preserve unique logical sequence order")
        node_events.append((seq, typed))
    pipeline_runs = _parse_pipeline_runs(
        row["pipeline_transition_runs"], candidate_descriptors,
        "pipeline_transition_runs", selector_indexes,
        protocol_version=protocol_version)
    # A full frame can include descriptors from this delivery only at initial
    # initialization; delta frames are checked against the retained registry.
    cursor_patches = _parse_cursor_patches(
        row["cursor_patches"], "cursor_patches", candidate_descriptors)
    resources = _parse_resources(
        row["active_resources"], candidate_descriptors, "active_resources",
        selector_indexes)
    return _ParsedEnvelope(
        canonical_bytes=header.canonical_bytes,
        replay_token=header.replay_token,
        producer=PRODUCER, epoch=header.epoch,
        delivery_sequence=delivery, snapshot_kind=header.snapshot_kind,
        tick=tick,
        base_watermark=header.base_watermark, seq_start=header.seq_start,
        seq_end=header.seq_end,
        event_count=header.event_count, base_version=header.base_version,
        result_version=header.result_version, protocol_version=protocol_version,
        descriptors=tuple(descriptors),
        selector_index_additions=MappingProxyType(selector_index_additions),
        dependency_index_additions=MappingProxyType(
            dependency_index_additions),
        node_events=tuple(node_events), pipeline_runs=pipeline_runs,
        cursor_patches=cursor_patches, resources=resources)


class _MaterializedNodeSource:
    """Offline node row source over pre-parsed ``(seq, typed event)`` pairs."""

    __slots__ = ("events", "index")

    def __init__(self, events):
        self.events = events
        self.index = 0

    @property
    def total(self):
        return len(self.events)

    def peek_seq(self):
        if self.index < len(self.events):
            return self.events[self.index][0]
        return None

    def take(self, work_transaction):
        seq, event = self.events[self.index]
        self.index += 1
        before = work_transaction.shadow._works.get(event.work_id)
        before_dependencies = (() if before is None else
                               tuple(before.dependencies))
        work_transaction.stage_event(seq, event)
        return (seq, event.kind, event.work_id, event.payload, before,
                before_dependencies)


class PipelineLedger:
    """Exact v3 archive descriptor/cursor/resource view.

    ``apply_envelope`` first validates an envelope into detached typed values.
    It does not mutate a WorkLedger; OnlineSchedulerBase coordinates its staged
    node-event transaction and this bounded state commit atomically.

    ``validate_envelope``/``_parse_envelope`` followed by ``prepare_envelope``
    over the materialized rows is the single decoder path: production, tests
    and offline archive comparison all share it.  (The v3 acceleration round
    trialled a streaming fused entry with zero per-event DTO materialization;
    it proved semantically identical but 13%-15% slower on real frames and
    was removed -- see online_scheduler_base._apply_work_ledger.)
    """

    PROTOCOL_VERSION = 3

    def __init__(self) -> None:
        self.initialized = False
        self.epoch: Optional[str] = None
        self.delivery_sequence = -1
        self.watermark = 0
        self.state_version = 0
        self.last_tick = -1
        self.last_replay_token: Optional[str] = None
        self._last_content = None
        self._descriptors = {}
        self._selector_indexes = {}
        self._dependency_endpoint_index = {}
        self._dependency_endpoint_index_by_rank = {}
        # Per-rank persistent interval roots for completed ordinary NodeStore
        # IDs and exact pipeline endpoint completions. Adjacent IDs coalesce,
        # so terminal proof survives delivery boundaries without retaining
        # released Work DTOs.
        self._completed_terminal_intervals_by_rank = {}
        self._unfinished_dependency_pins_by_segment = {}
        self._unfinished_dependency_waiters_by_endpoint = {}
        self._descriptor_state = {}
        self._active_segments = set()
        self._cursor_runs_by_segment = {}
        self._cursor_versions_by_segment = {}
        self._active_transfers_by_run = {}
        self._active_transfer_ids = set()
        self._active_transfer_id_counts = {}
        self._active_endpoint_counts = {}
        self._cursor_unissued_by_segment = {}
        self._source_release_watermarks = {}
        self._release_watermark_keys_by_domain = {}
        self._active_resources = ()
        self._slots_by_key = {}
        # V4-only private target-slot lineage. Entries survive row erasure so
        # a later incarnation cannot restart its predecessor chain.
        self._target_slot_lineage = {}
        # Bounded by ready shards in active target slots (at most two slots
        # per rank/domain), not by historical tiles. Each row carries its
        # exact current slot owner for later consume validation.
        self._target_ready_shards_by_slot = {}
        self._credits_by_key = {}
        self._obligations_by_id = {}
        self._slot_owner_counts = {}
        self._source_occupancy = {}
        self._target_occupancy = {}
        self._credit_ack_counts = {}
        self._ack_pins_by_segment = {}
        self._active_descriptor_count_by_domain = {}
        self._slot_row_counts_by_domain = {}
        self._credit_row_counts_by_domain = {}
        self._obligations_by_segment = {}
        self._obligation_remaining_by_segment = {}
        self.replay_count = 0
        self._snapshot_state_cache = None
        self._pricing_snapshot_cache = None

    def validate_envelope(self, envelope: Any) -> _ParsedEnvelope:
        """Parse a whole input without modifying accepted state."""
        is_last_replay = (
            issubclass(type(envelope), _RuntimeMapping)
            and type(envelope.get("delivery_sequence")) is int
            and envelope.get("delivery_sequence") == self.delivery_sequence
            and self.delivery_sequence >= 0)
        parsed = _parse_envelope(
            envelope, self._descriptors, self._selector_indexes,
            allow_retained_replay=is_last_replay,
            protocol_version=self.PROTOCOL_VERSION)
        is_last_replay = parsed.delivery_sequence == self.delivery_sequence
        for descriptor, _commit in parsed.descriptors:
            segment_id = descriptor.identity.segment_id
            prior = self._descriptors.get(segment_id)
            if prior is not None and not is_last_replay:
                raise PipelineLedgerError("descriptor segment was declared more than once")
        return parsed

    @staticmethod
    def _slot_key(row):
        return _slot_key(row)

    @staticmethod
    def _credit_key(row):
        return _credit_key(row)

    @staticmethod
    def _obligation_key(row):
        return _obligation_key(row)

    def prepare_envelope(self, envelope: Any, *,
                         work_transaction=None,
                         _parsed: Optional[_ParsedEnvelope] = None
                         ) -> PreparedPipelineUpdate:
        """Validate and COW-stage one globally ordered v3 delivery.

        ``validate_envelope``/``_parse_envelope`` materializes every node
        event DTO and parsed pipeline run first, then this method checks the
        global sequence and applies the staged transitions through the
        shared ``_stage_delivery`` core.
        """
        if _parsed is None:
            parsed = self.validate_envelope(envelope)
        elif isinstance(_parsed, _ParsedEnvelope):
            # Internal fast path for the scheduler: it parses once, keeps this
            # DTO local to the same synchronous call, then stages it here.
            # State/version/sequence checks below still run in full.
            parsed = _parsed
        else:
            raise PipelineLedgerError(
                "internal parsed envelope must be a validated pipeline DTO")
        replay = self._gate_delivery(parsed)
        if replay is not None:
            return replay

        if len(parsed.node_events) + len(parsed.pipeline_runs) != parsed.event_count:
            raise PipelineLedgerError(
                "logical event count differs from typed transition count")
        node_check = pipeline_check = 0
        expected_seq = parsed.seq_start
        previous_tick = self.last_tick
        while (node_check < len(parsed.node_events)
               or pipeline_check < len(parsed.pipeline_runs)):
            node = (parsed.node_events[node_check]
                    if node_check < len(parsed.node_events) else None)
            pipeline = (parsed.pipeline_runs[pipeline_check]
                        if pipeline_check < len(parsed.pipeline_runs) else None)
            node_seq = node[0] if node is not None else None
            pipeline_seq = (pipeline["first_logical_seq"]
                            if pipeline is not None else None)
            if node_seq is not None and node_seq == pipeline_seq:
                raise PipelineLedgerError(
                    "node and pipeline transitions share a logical sequence")
            take_node = (pipeline_seq is None or (node_seq is not None
                                                   and node_seq < pipeline_seq))
            if take_node:
                seq, event = node
                event_tick = event.tick
                node_check += 1
            else:
                seq = pipeline_seq
                event_tick = pipeline["tick_first"]
                pipeline_check += 1
            if seq != expected_seq:
                raise PipelineLedgerError(
                    "node/pipeline transitions do not form one global logical sequence")
            if event_tick < previous_tick or event_tick > parsed.tick:
                raise PipelineLedgerError(
                    "global node/pipeline transition ticks are not monotonic")
            previous_tick = event_tick
            expected_seq += 1
        if expected_seq != parsed.seq_end + 1:
            raise PipelineLedgerError(
                "logical event count does not cover the declared sequence interval")
        if parsed.node_events and work_transaction is None:
            raise PipelineLedgerError(
                "v3 node transitions require a coordinated WorkLedger COW transaction")

        return self._stage_delivery(
            parsed,
            work_transaction=work_transaction,
            node_source=_MaterializedNodeSource(parsed.node_events),
            pipeline_rows=parsed.pipeline_runs)

    def _replay_update(self, parsed: _ParsedEnvelope
                       ) -> PreparedPipelineUpdate:
        """Zero-change update for an idempotent replay of the last delivery."""
        empty = MappingProxyType({})
        return PreparedPipelineUpdate(
            parsed=parsed, descriptor_additions=empty,
            descriptor_state_updates=empty, descriptor_retirements=(),
            cursor_changes=empty, cursor_version_updates=empty,
            cursor_transfer_changes=empty, active_endpoint_deltas=empty,
            resources=self._active_resources, slot_changes=empty,
            target_slot_lineage_changes=empty,
            target_ready_changes=empty,
            terminal_provenance_updates=empty,
            credit_changes=empty, obligation_changes=empty,
            aggregate_deltas=empty,
            selector_index_additions=empty,
            dependency_index_additions=empty,
            dependency_rank_index_additions=empty,
            changed=False, replayed=True)

    def _gate_delivery(self, parsed: _ParsedEnvelope
                       ) -> Optional[PreparedPipelineUpdate]:
        """Shared delivery gates for both the offline and fused entries.

        Checks the decoder protocol match, the idempotent last-replay branch
        (returns the zero-change update), delivery continuity, full/delta
        preconditions and envelope tick monotonicity.  Returns None when the
        delivery may be applied.
        """
        if parsed.protocol_version != self.PROTOCOL_VERSION:
            raise PipelineLedgerError(
                "internal parsed envelope protocol does not match this ledger")
        if parsed.delivery_sequence == self.delivery_sequence and self.delivery_sequence >= 0:
            if parsed.epoch != self.epoch or parsed.canonical_bytes != self._last_content:
                raise PipelineLedgerError(
                    "last replay token carries typed-different canonical content")
            return self._replay_update(parsed)
        if parsed.delivery_sequence != self.delivery_sequence + 1:
            raise PipelineLedgerError("delivery_sequence gap or stale replay")
        if parsed.snapshot_kind == "full":
            if self.initialized or parsed.base_watermark != 0 or parsed.base_version != 0:
                raise PipelineLedgerError("full snapshot is only valid for initial state")
        else:
            if not self.initialized:
                raise PipelineLedgerError("delta received before initial full snapshot")
            if parsed.epoch != self.epoch:
                raise PipelineLedgerError("v3 epoch changed during run")
            if (parsed.base_watermark != self.watermark
                    or parsed.base_version != self.state_version):
                raise PipelineLedgerError("delta base watermark/state version mismatch")
        if parsed.tick < self.last_tick:
            raise PipelineLedgerError("v3 tick regressed")
        return None

    def _stage_delivery(self, parsed: _ParsedEnvelope, *,
                        work_transaction,
                        node_source,
                        pipeline_rows) -> PreparedPipelineUpdate:
        """Shared COW application core for the delivery entries.

        ``node_source`` supplies the staged node rows in logical order
        (the pre-materialized offline event tuples); ``pipeline_rows`` is
        any iterable of parsed pipeline runs.
        """
        additions = {}
        initial_state_updates = {}
        for descriptor, entry in parsed.descriptors:
            segment = descriptor.identity.segment_id
            if segment in self._descriptors or segment in additions:
                raise PipelineLedgerError(
                    "descriptor declaration conflicts with retained descriptor")
            additions[segment] = descriptor
            initial_state_updates[segment] = {
                "commit_tick": entry["commit_tick"],
                "service_done": entry["service_done"],
                "settle_done": entry["settle_done"],
                "retired": entry["retired"],
                "request_closed": False,
                "cursor_version": entry["cursor_version"],
            }
        descriptors = ChainMap(additions, self._descriptors)
        states = ChainMap(initial_state_updates, self._descriptor_state)
        dependency_root_changes = {}
        dependency_rank_root_changes = {}
        for dependency_index in parsed.dependency_index_additions.values():
            for prefix, intervals in dependency_index.intervals_by_prefix.items():
                root = dependency_root_changes.get(
                    prefix, self._dependency_endpoint_index.get(prefix))
                for interval in intervals:
                    root = _dependency_interval_insert(root, interval)
                dependency_root_changes[prefix] = root
            for rank, intervals in dependency_index.intervals_by_rank.items():
                root = dependency_rank_root_changes.get(
                    rank, self._dependency_endpoint_index_by_rank.get(rank))
                for interval in intervals:
                    root = _dependency_interval_insert(root, interval)
                dependency_rank_root_changes[rank] = root
        dependency_view = ChainMap(
            dependency_root_changes, self._dependency_endpoint_index)
        dependency_rank_view = ChainMap(
            dependency_rank_root_changes,
            self._dependency_endpoint_index_by_rank)

        slot_overlay = _RowOverlay(self._slots_by_key)
        target_worksets = {}
        target_slot_lineage_changes = {}
        target_ready_overlay = _RowOverlay(
            self._target_ready_shards_by_slot)
        credit_overlay = _RowOverlay(self._credits_by_key)
        obligation_overlay = _RowOverlay(self._obligations_by_id)
        slot_owner_counts = _CountOverlay(self._slot_owner_counts)
        source_occupancy = _CountOverlay(self._source_occupancy)
        target_occupancy = _CountOverlay(self._target_occupancy)
        credit_ack_domains = _CountOverlay(self._credit_ack_counts)
        ack_pins_by_segment = _CountOverlay(self._ack_pins_by_segment)
        active_domain_counts = _CountOverlay(
            self._active_descriptor_count_by_domain)
        slot_row_counts = _CountOverlay(self._slot_row_counts_by_domain)
        credit_row_counts = _CountOverlay(self._credit_row_counts_by_domain)
        active_transfer_id_counts = _CountOverlay(self._active_transfer_id_counts)
        obligation_remaining = _CountOverlay(
            self._obligation_remaining_by_segment)
        dependency_pins = _CountOverlay(
            self._unfinished_dependency_pins_by_segment)
        dependency_waiters = _CountOverlay(
            self._unfinished_dependency_waiters_by_endpoint)
        cursor_unissued = _CountOverlay(self._cursor_unissued_by_segment)
        release_watermarks = _CountOverlay(self._source_release_watermarks)

        cursor_versions = dict()
        for segment, entry in initial_state_updates.items():
            cursor_versions[segment] = entry["cursor_version"]
        touched_state = {}
        retirements = []
        descriptor_commits = set()
        descriptor_transition_counts = {}
        cursor_dirty = set()
        cursor_sims = {}
        cursor_changes = {}
        cursor_version_updates = {}
        cursor_transfer_changes = {}
        active_endpoint_deltas = {}
        cursor_active_deltas = {}
        expected_obligation_rows = {}
        resource_counts = {}
        domain_retirements = set()
        for resource in parsed.resources:
            segment = resource["descriptor_segment_id"]
            if segment is not None:
                resource_counts[segment] = resource_counts.get(segment, 0) + 1
        parsed_cursor_patches = {
            row["segment_id"]: row for row in parsed.cursor_patches
        }
        parsed_cursor_patch_run_indexes = {
            row["segment_id"]: frozenset(
                run_index for run_index, _value in row["runs"])
            for row in parsed.cursor_patches
        }
        node_terminal_frontier = set()
        terminal_provenance_roots = {}
        pending_ready_after_clear = set()

        def current_state(segment):
            state = touched_state.get(segment)
            if state is not None:
                return state
            raw = states.get(segment)
            if raw is None:
                raise PipelineLedgerError(
                    f"transition references unknown descriptor {segment!r}")
            state = dict(raw)
            touched_state[segment] = state
            return state

        def _active_state(work):
            return (work is not None and work.state in
                    ("committed", "ready", "issued", "in_service"))

        def _terminal_proven(rank, node_id):
            return ((rank, node_id) in node_terminal_frontier
                    or _completed_interval_contains(
                        terminal_provenance_roots.get(
                            rank,
                            self._completed_terminal_intervals_by_rank.get(
                                rank)), node_id))

        def _record_dependency_terminal(rank, node_id, segments=None):
            root = terminal_provenance_roots.get(
                rank, self._completed_terminal_intervals_by_rank.get(rank))
            terminal_provenance_roots[rank] = _completed_interval_add(
                root, node_id)
            if segments is None:
                segments = _dependency_segments_for_rank_node(
                    rank, node_id, dependency_rank_view)
            for segment in segments:
                key = (rank, node_id, segment)
                waiters = dependency_waiters.get(key)
                if waiters:
                    dependency_waiters.add(key, -waiters)
                    dependency_pins.add(segment, -waiters)
                    if dependency_waiters.get(key) < 0:
                        raise PipelineLedgerError(
                            "ordinary dependency waiter aggregate underflow")
                    if dependency_pins.get(segment) < 0:
                        raise PipelineLedgerError(
                            "unfinished dependency pin aggregate underflow")

        def stage_node_events_before(logical_seq):
            # The node source (pre-materialized offline events or the fused
            # streaming validator) owns row validation and WorkLedger staging;
            # this closure keeps only the pipeline-side bookkeeping that must
            # interleave at each event's exact global-sequence position.
            while True:
                next_seq = node_source.peek_seq()
                if next_seq is None or next_seq >= logical_seq:
                    return
                (seq, kind, work_id, payload,
                 before, before_dependencies) = node_source.take(
                    work_transaction)
                was_active = _active_state(before)
                after = work_transaction.shadow._works.get(work_id)
                if kind in ("completion", "release"):
                    rank = payload.get("rank")
                    node_id = payload.get("node_id")
                    if type(rank) is int and type(node_id) is int:
                        # NodeStore ids are producer-epoch/rank scoped. The
                        # descriptor's affine JSON id is zero-based, while
                        # the ordinary node store id is JSON id + 1.
                        node_terminal_frontier.add((rank, node_id))
                        _record_dependency_terminal(rank, node_id)
                is_active = _active_state(after)
                if was_active != is_active:
                    direction = -1 if was_active else 1
                    dependencies = (before_dependencies if was_active else
                                    tuple(after.dependencies))
                    for dependency in dependencies:
                        reference = _dependency_reference(
                            dependency, dependency_view)
                        if reference is None:
                            continue
                        segment, dependency_rank, dependency_node_id = reference
                        state = touched_state.get(segment, states.get(segment))
                        if state is None:
                            raise PipelineLedgerError(
                                "ordinary dependency references an unknown pipeline descriptor")
                        if _terminal_proven(dependency_rank,
                                            dependency_node_id):
                            if state["retired"]:
                                # A retired descriptor may retain a compact
                                # terminal proof for endpoints that really
                                # completed. This lets a later ordinary graph
                                # commit refer to the immutable completed
                                # parent without pinning retired cursor rows.
                                continue
                            continue
                        if state["retired"]:
                            raise PipelineLedgerError(
                                "ordinary dependency references an incomplete endpoint in a retired pipeline descriptor")
                        precommit_declaration = (
                            kind == "commit"
                            and segment in additions
                            and segment not in descriptor_commits)
                        if ((segment not in self._active_segments
                             and segment not in descriptor_commits
                             and not precommit_declaration)
                                or state is None):
                            raise PipelineLedgerError(
                                "ordinary dependency references an uncommitted pipeline descriptor")
                        waiter_key = (
                            dependency_rank, dependency_node_id, segment)
                        if direction < 0 and dependency_waiters.get(
                                waiter_key) <= 0:
                            raise PipelineLedgerError(
                                "unfinished ordinary dependency waiter underflow")
                        dependency_waiters.add(waiter_key, direction)
                        dependency_pins.add(segment, direction)
                        if dependency_pins.get(segment) < 0:
                            raise PipelineLedgerError(
                                "unfinished dependency pin aggregate underflow")

        def seed_cursor_progress(segment, run_index):
            key = (segment, run_index)
            if key in cursor_sims:
                return cursor_sims[key]
            descriptor = descriptors.get(segment)
            if descriptor is None or run_index >= len(descriptor.endpoint_runs):
                raise PipelineLedgerError("cursor run references unknown descriptor extent")
            if segment in additions:
                row = _initial_cursor_run(descriptor, segment, run_index)
            else:
                row = self._cursor_runs_by_segment.get(segment, {}).get(run_index)
                if row is None:
                    raise PipelineLedgerError(
                        "active descriptor cursor run is absent from accepted state")
            original_ooo = tuple(
                (item["first_ordinal"], item["end_ordinal_exclusive"])
                for item in row["completed_out_of_order"])
            completed_ooo, completed_ooo_count = (
                _completed_interval_build_from_half_open(
                    original_ooo))
            run = descriptor.endpoint_runs[run_index]
            active = {}
            for endpoint in row["active_endpoints"]:
                local_tile = endpoint["tile_ordinal"] - run.first_tile
                shard_index = run.shard_ordinals.index(endpoint["shard_ordinal"])
                leg_index = next(
                    index for index, leg in enumerate(run.legs)
                    if leg.endpoint == endpoint["endpoint"])
                ordinal = (local_tile * len(run.shard_ordinals) * len(run.legs)
                           + shard_index * len(run.legs) + leg_index)
                active[endpoint["store_id"]] = (ordinal, endpoint)
            state = current_state(segment)
            progress = {
                "row": row,
                "run": run,
                "total": _run_endpoint_total(run),
                "completed": row["completed_endpoint_count"],
                "completed_total": (
                    row["completed_endpoint_count"] + completed_ooo_count),
                "cancelled": row["cancelled_endpoint_count"],
                "remaining": row["remaining_endpoint_count"],
                "ooo_root": completed_ooo,
                # The treap normalizes adjacent rows for future interval
                # updates. Preserve the input spelling until a completion
                # actually changes this set, matching the prior list path for
                # issue-only cursor patches.
                "ooo_wire_ranges": original_ooo,
                "ooo_changed": False,
                "active": active,
                "position": _cursor_position_ordinal(row, run),
                "closed": state.get("request_closed", False),
            }
            cursor_sims[key] = progress
            return progress

        def progress_has_completed(progress, ordinal):
            return (ordinal < progress["completed"]
                    or _completed_interval_contains(
                        progress["ooo_root"], ordinal))

        def selector_index_for(segment):
            return (self._selector_indexes.get(segment)
                    or parsed.selector_index_additions.get(segment))

        def endpoint_completion_proven(segment, descriptor, rank, group,
                                       tile, shard, endpoint):
            indexes = selector_index_for(segment)
            if indexes is None:
                return False
            match = _descriptor_selector_at(
                indexes, group, rank, tile, shard, endpoint)
            if match is None:
                return False
            run_index, leg_index = match
            run = descriptor.endpoint_runs[run_index]
            try:
                shard_position = run.shard_ordinals.index(shard)
            except ValueError:
                return False
            local_tile = tile - run.first_tile
            if local_tile < 0 or local_tile >= run.tile_count:
                return False
            ordinal = (local_tile * len(run.shard_ordinals) * len(run.legs)
                       + shard_position * len(run.legs) + leg_index)
            return progress_has_completed(
                seed_cursor_progress(segment, run_index), ordinal)

        def endpoint_store_id_for(segment, descriptor, rank, group, tile,
                                  shard, endpoint):
            indexes = selector_index_for(segment)
            if indexes is None:
                return None
            match = _descriptor_selector_at(
                indexes, group, rank, tile, shard, endpoint)
            if match is None:
                return None
            run_index, leg_index = match
            return descriptor.endpoint_id(
                run_index, tile, shard, leg_index) + 1

        def consume_terminal_proven(descriptor, segment, rank, group,
                                    tile, shard):
            indexes = selector_index_for(segment)
            if indexes is None:
                return False
            match = _consume_completion_at(
                indexes, rank, group, tile, shard)
            if match is None:
                return False
            completion_index, shard_position = match
            completion = descriptor.consume_completion_runs[completion_index]
            dependency_json_id = (
                completion.first_dependency_json_id
                + (tile - completion.first_tile)
                * completion.dependency_tile_stride
                + shard_position * completion.dependency_shard_stride)
            if completion.dependency_kind == "ordinary_node_terminal":
                store_id = dependency_json_id + 1
                return ((rank, store_id) in node_terminal_frontier
                        or _completed_interval_contains(
                            terminal_provenance_roots.get(
                                rank,
                                self._completed_terminal_intervals_by_rank.get(
                                    rank)), store_id))
            if completion.dependency_kind == "virtual_endpoint_terminal":
                marker = _descriptor_selector_at(
                    indexes, group, rank, tile, shard, "body_marker")
                if marker is None:
                    return False
                run_index, leg_index = marker
                if descriptor.endpoint_id(
                        run_index, tile, shard, leg_index) != dependency_json_id:
                    return False
                return endpoint_completion_proven(
                    segment, descriptor, rank, group, tile, shard,
                    "body_marker")
            return False

        def _completed_count(progress):
            return progress["completed_total"]

        def _first_pending_store(descriptor, run_index, progress):
            if progress["closed"]:
                if not progress["active"]:
                    return None
                first_store = min(progress["active"])
                return first_store
            ordinal = progress["completed"]
            if ordinal >= progress["total"]:
                return None
            run = descriptor.endpoint_runs[run_index]
            per_tile = len(run.shard_ordinals) * len(run.legs)
            tile_offset, within = divmod(ordinal, per_tile)
            shard_index, leg_index = divmod(within, len(run.legs))
            return descriptor.endpoint_id(
                run_index, run.first_tile + tile_offset,
                run.shard_ordinals[shard_index], leg_index) + 1

        def expected_obligation(descriptor, segment, run_index, progress):
            remaining = (len(progress["active"]) if progress["closed"]
                         else progress["total"] - _completed_count(progress)
                         - progress["cancelled"])
            if remaining <= 0:
                return None
            run = descriptor.endpoint_runs[run_index]
            first_store_id = _first_pending_store(
                descriptor, run_index, progress)
            if first_store_id is None:
                raise PipelineLedgerError(
                    "unfinished cursor run has no first unresolved endpoint")
            return {
                "obligation_id": f"{segment}/run/{run_index}",
                "segment_id": segment,
                "rank": run.rank,
                "first_store_id": first_store_id,
                "remaining_endpoint_count": remaining,
                "state": "active" if progress["active"] else "pending",
            }

        def merge_completion(progress, ordinal, path, segment, run_index):
            merge_completion_range(progress, ordinal, ordinal + 1,
                                   path, segment, run_index)

        def merge_completion_range(progress, first, end, path,
                                   segment, run_index):
            try:
                root, completed = _completed_interval_add_range(
                    progress["ooo_root"], progress["completed"],
                    progress["total"], first, end)
            except PipelineLedgerError as exc:
                raise PipelineLedgerError(
                    f"{path} completes an endpoint twice") from exc
            completed_total = progress["completed_total"] + end - first
            if completed_total > progress["total"]:
                raise PipelineLedgerError(
                    f"{path} completed count exceeds its cursor extent")
            progress["ooo_root"] = root
            progress["completed"] = completed
            progress["completed_total"] = completed_total
            progress["ooo_changed"] = True
            cursor_dirty.add((segment, run_index))

        def advance_inactive_positions(descriptor, progress, segment,
                                       run_index, run_transition):
            """Skip only the current tile's bounded axes, then jump over the
            descriptor-guaranteed inactive suffix arithmetically."""
            if progress["position"] >= progress["total"]:
                return
            run = progress["run"]
            shard_count, leg_count = len(run.shard_ordinals), len(run.legs)
            per_tile = shard_count * leg_count
            tile_offset = progress["position"] // per_tile
            tile_start = tile_offset * per_tile
            tile_end = min(progress["total"], tile_start + per_tile)
            skipped_start = progress["position"]
            transition_selector = run_transition["endpoint_selector"]
            transition_shard_facts = run_transition["_selected_shard_tile"]
            while progress["position"] < tile_end:
                within = progress["position"] - tile_start
                shard_index, leg_index = divmod(within, leg_count)
                shard_ordinal = run.shard_ordinals[shard_index]
                leg = run.legs[leg_index]
                is_body_marker = leg.endpoint == "body_marker"
                if is_body_marker:
                    is_active = True
                else:
                    candidate_tile_ordinal = run.first_tile + tile_offset
                    if (transition_selector is not None
                            and transition_shard_facts is not None
                            and transition_selector["group_ordinal"]
                                == run.group_ordinal
                            and transition_selector["tile_ordinal"]
                                == candidate_tile_ordinal
                            and transition_selector["shard_ordinal"]
                                == shard_ordinal):
                        candidate_shard = transition_shard_facts
                    else:
                        _candidate_global_ordinal, candidate_shard = (
                            descriptor.selected_shard_tile(
                                run.group_ordinal,
                                run.first_tile + tile_offset - 1,
                                shard_ordinal))
                    is_active = candidate_shard.emit_endpoint
                if is_active:
                    if progress["position"] > skipped_start:
                        merge_completion_range(
                            progress, skipped_start, progress["position"],
                            "inactive cursor skip", segment, run_index)
                        cursor_unissued.add(
                            segment, -(progress["position"] - skipped_start))
                    return
                progress["position"] += 1
            body_positions = [
                (shard_index, leg_index)
                for shard_index in range(shard_count)
                for leg_index, leg in enumerate(run.legs)
                if leg.endpoint == "body_marker"]
            if body_positions and tile_offset + 1 < run.tile_count:
                shard_index, leg_index = body_positions[0]
                next_position = ((tile_offset + 1) * per_tile
                                 + shard_index * leg_count + leg_index)
                progress["position"] = next_position
            else:
                progress["position"] = progress["total"]
                next_position = progress["position"]
            if next_position > skipped_start:
                merge_completion_range(
                    progress, skipped_start, next_position,
                    "inactive cursor skip", segment, run_index)
                cursor_unissued.add(segment, -(next_position - skipped_start))

        def selected_slot_and_credit(descriptor, transition):
            selector = transition["endpoint_selector"]
            if selector is None:
                return None
            run_index = transition["endpoint_run_index"]
            run = descriptor.endpoint_runs[run_index]
            return (
                run, transition["_selected_global_tile_ordinal"],
                transition["_selected_shard_tile"],
                run.legs[transition["endpoint_leg_index"]].endpoint)

        def add_row_aggregate(row, sign):
            if row is None:
                return
            slot_row_counts.add(row["buffer_domain_id"], sign)
            if row["occupied"]:
                slot_owner_counts.add(row["owner"]["segment_id"], sign)
                if row["role"] == "source":
                    source_occupancy.add(_credit_key({
                        "buffer_domain_id": row["buffer_domain_id"],
                        "source_rank": row["rank"],
                        "shard_ordinal": row["shard_ordinal"],
                    }), sign)
                else:
                    target_key = (row["buffer_domain_id"], row["rank"])
                    target_occupancy.add(target_key, sign)

        def target_workset_for(key):
            workset = target_worksets.get(key)
            if workset is not None:
                return workset
            row = slot_overlay.get(key)
            if row is None:
                return None
            if row["role"] != "target":
                raise PipelineLedgerError("target delta addresses a source slot")
            consumed = _copy_consumed_slot_members(
                row["consumed_target_shards"])
            workset = {
                "row": dict(row),
                "consumed": consumed,
                "remaining": len(row["target_shards_to_consume"])
                    - len(consumed),
                "non_target_count": 0,
            }
            target_worksets[key] = workset
            return workset

        def current_target_row(key):
            workset = target_worksets.get(key)
            if workset is not None:
                return workset["row"]
            return slot_overlay.get(key)

        def _target_row_empty(key):
            return {
                "buffer_domain_id": key[0], "role": "target",
                "rank": key[2], "shard_ordinal": None,
                "slot_index": key[4], "occupied": False,
                "owner": None, "consumed": False,
                "target_shards_to_consume": (),
                "consumed_target_shards": (),
            }

        def _target_lineage(key):
            return target_slot_lineage_changes.get(
                key, self._target_slot_lineage.get(key, (0, 0, 0)))

        def _claim_target_members(owner, descriptor, key, declared):
            owner_descriptor = descriptors.get(owner["segment_id"])
            if owner_descriptor is None:
                raise PipelineLedgerError(
                    "target claim owner references an unknown descriptor")
            ident = owner_descriptor.identity
            if (ident.buffer_domain_id != key[0]
                    or ident.rank != key[2]
                    or ident.generation != owner["pipeline_generation"]
                    or ident.stage_epoch != owner["stage_epoch"]
                    or ident.staging_allocation_id
                        != owner["staging_allocation_id"]):
                raise PipelineLedgerError(
                    "target claim owner disagrees with descriptor identity")
            owner_index = selector_index_for(owner["segment_id"])
            group = (None if owner_index is None else
                     owner_index.groups_by_ordinal.get(owner["group_ordinal"]))
            if group is None:
                raise PipelineLedgerError(
                    "target claim owner references an unknown tile group")
            if (owner["tile_ordinal"] < 1
                    or owner["tile_ordinal"] > group.tile_axis.tile_count):
                raise PipelineLedgerError(
                    "target claim owner points outside descriptor tile range")
            global_tile = (group.tile_axis.global_tile_first
                           + owner["tile_ordinal"] - 1)
            if key[4] != (global_tile - 1) % 2:
                raise PipelineLedgerError(
                    "target claim slot index disagrees with global tile parity")
            expected = []
            for shard_ordinal in range(len(group.shards)):
                _global, shard = owner_descriptor.selected_shard_tile(
                    owner["group_ordinal"], owner["tile_ordinal"] - 1,
                    shard_ordinal)
                if shard.emit_endpoint and shard.target_rank == ident.rank:
                    expected.append(shard_ordinal)
            expected = tuple(expected)
            if not expected or declared != expected:
                raise PipelineLedgerError(
                    "target claim does not declare the exact descriptor membership")
            return expected

        def _consume_delta_members(workset, owner_descriptor, segment,
                                   cause):
            run_index = cause["consume_run_index"]
            if run_index >= len(owner_descriptor.consume_completion_runs):
                raise PipelineLedgerError(
                    "target consume cause run index is outside the descriptor")
            completion = owner_descriptor.consume_completion_runs[run_index]
            row = workset["row"]
            owner = row["owner"]
            tile = owner["tile_ordinal"]
            if (completion.rank != cause["terminal_rank"]
                    or completion.rank != row["rank"]
                    or completion.group_ordinal != owner["group_ordinal"]
                    or tile < completion.first_tile
                    or tile >= completion.first_tile + completion.tile_count):
                raise PipelineLedgerError(
                    "target consume cause run does not cover the slot owner")
            tile_delta = tile - completion.first_tile
            dependency_json_id = (
                completion.first_dependency_json_id
                + tile_delta * completion.dependency_tile_stride)
            terminal_store_id = dependency_json_id + 1
            if terminal_store_id > U64_MAX:
                raise PipelineLedgerError(
                    "target consume cause affine terminal ID overflows uint64")
            if completion.dependency_shard_stride:
                if cause["terminal_store_id"] < terminal_store_id:
                    raise PipelineLedgerError(
                        "target consume cause store ID precedes its affine base")
                difference = cause["terminal_store_id"] - terminal_store_id
                if difference % completion.dependency_shard_stride:
                    raise PipelineLedgerError(
                        "target consume cause store ID is outside its affine shard mapping")
                shard_position = difference // completion.dependency_shard_stride
                if shard_position >= len(completion.shard_ordinals):
                    raise PipelineLedgerError(
                        "target consume cause shard index is outside its run")
                mapped = (completion.shard_ordinals[shard_position],)
            else:
                if cause["terminal_store_id"] != terminal_store_id:
                    raise PipelineLedgerError(
                        "target consume cause store ID differs from its shared marker")
                mapped = tuple(completion.shard_ordinals)
            indexes = selector_index_for(segment)
            if indexes is None:
                raise PipelineLedgerError(
                    "target consume cause has no retained descriptor index")
            for shard_ordinal in mapped:
                proof = _consume_completion_at(
                    indexes, cause["terminal_rank"],
                    completion.group_ordinal, tile, shard_ordinal)
                if proof is None or proof[0] != run_index:
                    raise PipelineLedgerError(
                        "target consume cause does not name the owner's terminal run")
                expected_json_id = (
                    completion.first_dependency_json_id
                    + tile_delta * completion.dependency_tile_stride
                    + proof[1] * completion.dependency_shard_stride)
                if expected_json_id + 1 != cause["terminal_store_id"]:
                    raise PipelineLedgerError(
                        "target consume cause store ID disagrees with its run member")
                if not consume_terminal_proven(
                        owner_descriptor, segment, cause["terminal_rank"],
                        completion.group_ordinal, tile, shard_ordinal):
                    raise PipelineLedgerError(
                        "target consume cause lacks prior terminal proof")
            expected_new = tuple(sorted(
                shard for shard in mapped
                if shard not in workset["consumed"]))
            return expected_new

        def _prove_target_consume_clear(row, owner_descriptor, selected,
                                        transition):
            owner = row["owner"]
            owner_segment = owner["segment_id"]
            owner_shards = row["target_shards_to_consume"]
            selected_consumer_proof = selected is None
            if selected is not None:
                run, _unused_global, shard, endpoint = selected
                selector = transition["endpoint_selector"]
                selected_consumer_proof = (
                    endpoint == "body_marker"
                    and run.rank == row["rank"]
                    and run.group_ordinal == owner["group_ordinal"]
                    and selector["tile_ordinal"] == owner["tile_ordinal"]
                    and shard.shard_ordinal in owner_shards
                    and endpoint_completion_proven(
                        owner_segment, owner_descriptor, row["rank"],
                        owner["group_ordinal"], owner["tile_ordinal"],
                        shard.shard_ordinal, "body_marker"))
            ready_and_terminal = bool(owner_shards)
            ready_tag = _target_owner_tag(owner)
            for target_shard in owner_shards:
                if not endpoint_completion_proven(
                        owner_segment, owner_descriptor, row["rank"],
                        owner["group_ordinal"], owner["tile_ordinal"],
                        target_shard, "data_recv"):
                    ready_and_terminal = False
                    break
                ready_key = (row["buffer_domain_id"], row["rank"],
                             row["slot_index"], target_shard)
                already_ready = target_ready_overlay.get(ready_key) == ready_tag
                if not already_ready:
                    pending_ready_after_clear.add((ready_key, ready_tag))
                if not consume_terminal_proven(
                        owner_descriptor, owner_segment, row["rank"],
                        owner["group_ordinal"], owner["tile_ordinal"],
                        target_shard):
                    ready_and_terminal = False
                    break
            return (bool(owner_shards) and selected_consumer_proof
                    and ready_and_terminal)

        def apply_target_delta(transition, descriptor, selected, delta):
            key = delta["key"]
            if key[0] != descriptor.identity.buffer_domain_id:
                raise PipelineLedgerError(
                    "target delta crosses the transition buffer domain")
            if key[2] != descriptor.identity.rank:
                raise PipelineLedgerError(
                    "target delta rank differs from its buffer-domain descriptor")
            before_generation, predecessor_seq, predecessor_count = _target_lineage(key)
            if (delta["before_slot_generation"] != before_generation
                    or delta["predecessor_logical_seq"] != predecessor_seq
                    or delta["predecessor_consumed_count"] != predecessor_count
                    or predecessor_seq >= transition["first_logical_seq"]):
                raise PipelineLedgerError(
                    "target delta predecessor lineage differs from accepted state")
            workset = target_workset_for(key)
            original_row = None if workset is None else workset["row"]
            initial_present = original_row is not None
            initial_occupied = bool(initial_present and original_row["occupied"])
            initial_owner = (original_row["owner"] if initial_occupied else None)
            initial_targets = (original_row["target_shards_to_consume"]
                               if initial_present else ())
            generation = before_generation
            if workset is None:
                workset = {"row": None, "consumed": set(),
                           "remaining": 0, "non_target_count": 0}
                target_worksets[key] = workset
            row = workset["row"]
            for op in delta["operations"]:
                op_kind = op["op"]
                op_generation = op["slot_generation"]
                if op_kind == "claim":
                    if generation == U64_MAX:
                        raise PipelineLedgerError(
                            "target slot generation overflows uint64")
                    if op_generation != generation + 1 or row is None or row["occupied"]:
                        raise PipelineLedgerError(
                            "target claim generation or empty-slot precondition differs")
                    owner = op["owner"]
                    owner_segment = owner["segment_id"]
                    if owner_segment != transition["descriptor_segment_id"]:
                        raise PipelineLedgerError(
                            "target claim assigns a different descriptor owner")
                    owner_descriptor = descriptors[owner_segment]
                    expected_targets = _claim_target_members(
                        owner, descriptor, key, op["target_shards"])
                    if transition["transition_kind"] not in (
                            "tile_issue", "consume_start", "fill_ack_progress",
                            "release_ack_progress", "consume_done"):
                        raise PipelineLedgerError(
                            "target slot ownership changed outside an issue/consume transition")
                    if selected is not None:
                        run, _global, shard, _endpoint = selected
                        selector = transition["endpoint_selector"]
                        if (owner["tile_ordinal"] != selector["tile_ordinal"]
                                or owner["group_ordinal"] != run.group_ordinal
                                or shard.target_rank != key[2]):
                            raise PipelineLedgerError(
                                "target claim owner differs from selected endpoint")
                    row = {
                        "buffer_domain_id": key[0], "role": "target",
                        "rank": key[2], "shard_ordinal": None,
                        "slot_index": key[4], "occupied": True,
                        "owner": owner, "consumed": False,
                        "target_shards_to_consume": op["target_shards"],
                        "consumed_target_shards": (),
                    }
                    workset["row"] = row
                    workset["consumed"].clear()
                    workset["remaining"] = len(op["target_shards"])
                    workset["non_target_count"] = 0
                    generation = op_generation
                elif op_kind == "initialize":
                    if op_generation != generation or row is not None:
                        raise PipelineLedgerError(
                            "target initialize generation or absent-row precondition differs")
                    if (transition["transition_kind"] != "descriptor_commit"
                            or descriptor.identity.buffer_domain_id != key[0]
                            or descriptor.identity.rank != key[2]):
                        raise PipelineLedgerError(
                            "target initialize is outside its descriptor commit")
                    row = _target_row_empty(key)
                    workset["row"] = row
                    workset["consumed"].clear()
                    workset["remaining"] = 0
                    workset["non_target_count"] = 0
                elif op_kind == "consume":
                    if (op_generation != generation or row is None
                            or not row["occupied"]
                            or transition["transition_kind"] != "consume_done"):
                        raise PipelineLedgerError(
                            "target consume lacks its occupied owner row")
                    owner = row["owner"]
                    owner_segment = owner["segment_id"]
                    if owner_segment != transition["descriptor_segment_id"]:
                        raise PipelineLedgerError(
                            "target consume owner differs from its sequenced descriptor")
                    owner_descriptor = descriptors[owner_segment]
                    expected_new = _consume_delta_members(
                        workset, owner_descriptor, owner_segment,
                        op["cause"])
                    if not expected_new or op["add_shards"] != expected_new:
                        raise PipelineLedgerError(
                            "target consume additions differ from exact cause mapping")
                    for shard_ordinal in expected_new:
                        workset["consumed"].add(shard_ordinal)
                        if _sorted_uint_tuple_contains(
                                row["target_shards_to_consume"], shard_ordinal):
                            workset["remaining"] -= 1
                        else:
                            workset["non_target_count"] += 1
                    row["consumed"] = bool(workset["consumed"])
                    if op["cleared"] != (workset["remaining"] == 0):
                        raise PipelineLedgerError(
                            "target consume cleared flag differs from remaining membership")
                    if op["cleared"]:
                        if not _prove_target_consume_clear(
                                row, owner_descriptor, selected, transition):
                            raise PipelineLedgerError(
                                "target consume cleared before its ready/terminal proof")
                        row = _target_row_empty(key)
                        workset["row"] = row
                        workset["consumed"].clear()
                        workset["remaining"] = 0
                        workset["non_target_count"] = 0
                elif op_kind == "request_close_clear":
                    if op_generation != generation or row is None or not row["occupied"]:
                        raise PipelineLedgerError(
                            "request-close clear lacks its occupied target row")
                    owner = row["owner"]
                    owner_descriptor = descriptors[owner["segment_id"]]
                    if (transition["transition_kind"] != "request_close"
                            or owner_descriptor.identity.request_id
                                != descriptor.identity.request_id
                            or owner_descriptor.identity.buffer_domain_id
                                != descriptor.identity.buffer_domain_id):
                        raise PipelineLedgerError(
                            "request-close clear touches a foreign request/domain")
                    row = _target_row_empty(key)
                    workset["row"] = row
                    workset["consumed"].clear()
                    workset["remaining"] = 0
                    workset["non_target_count"] = 0
                elif op_kind == "erase":
                    if (op_generation != generation or row is None
                            or row["occupied"]
                            or transition["transition_kind"] != "descriptor_retire"):
                        raise PipelineLedgerError(
                            "target erase requires an empty row at descriptor retirement")
                    row = None
                    workset["row"] = None
                    workset["consumed"].clear()
                    workset["remaining"] = 0
                    workset["non_target_count"] = 0
            if row is not None and row["occupied"]:
                if (workset["non_target_count"] != 0
                        or len(workset["consumed"]) + workset["remaining"]
                            != len(row["target_shards_to_consume"])
                        or row["consumed"] != bool(workset["consumed"])):
                    raise PipelineLedgerError(
                        "occupied target row violates its full-row consume invariant")
            final_present = row is not None
            final_occupied = bool(final_present and row["occupied"])
            final_owner = row["owner"] if final_occupied else None
            # Public row/owner aggregates are updated once per logical target
            # patch, never once per operation inside its coalesced delta.
            if initial_present:
                slot_row_counts.add(key[0], -1)
            if initial_occupied:
                slot_owner_counts.add(initial_owner["segment_id"], -1)
                target_occupancy.add((key[0], key[2]), -1)
            if final_present:
                slot_row_counts.add(key[0], 1)
            if final_occupied:
                slot_owner_counts.add(final_owner["segment_id"], 1)
                target_occupancy.add((key[0], key[2]), 1)
            old_tag = (None if not initial_occupied else
                       _target_owner_tag(initial_owner))
            new_tag = (None if not final_occupied else
                       _target_owner_tag(final_owner))
            if old_tag is not None and (new_tag is None or old_tag != new_tag):
                for target_shard in initial_targets:
                    target_ready_overlay.delete(
                        (key[0], key[2], key[4], target_shard))
            # Preserve the old occupied-owner displacement and slot-reuse
            # proof when a coalesced patch keeps both rows occupied.
            if initial_occupied and final_occupied:
                if initial_owner["segment_id"] != final_owner["segment_id"]:
                    old_state = touched_state.get(
                        initial_owner["segment_id"],
                        states.get(initial_owner["segment_id"]))
                    if old_state is None or not (
                            old_state["settle_done"] or old_state["retired"]):
                        raise PipelineLedgerError(
                            "foreign target slot owner was displaced before settle")
                elif (initial_owner["group_ordinal"] != final_owner["group_ordinal"]
                      or initial_owner["tile_ordinal"] != final_owner["tile_ordinal"]):
                    raise PipelineLedgerError(
                        "target slot owner changed before its legal clear")
            final_count = (len(workset["consumed"])
                           if final_occupied else 0)
            target_slot_lineage_changes[key] = (
                generation, transition["first_logical_seq"], final_count)
            touched_targets = {(key[0], key[2])}
            return touched_targets

        def apply_slot_patches(transition, descriptor, selected):
            touched_keys = set()
            touched_targets = set()
            target_deltas = []
            if parsed.protocol_version == 4:
                source_patches = []
                for patch in transition["slot_patches"]:
                    if isinstance(patch, dict) and patch.get("form") == "target_delta":
                        target_deltas.append(patch)
                    else:
                        source_patches.append(patch[1:])
            else:
                source_patches = transition["slot_patches"]
            for key, before, after in source_patches:
                present = slot_overlay.has(key)
                current = slot_overlay.get(key)
                if before is None:
                    if present:
                        raise PipelineLedgerError(
                            "slot patch expected an absent before row")
                elif not present or not _typed_equal(current, before):
                    raise PipelineLedgerError(
                        "slot patch before row differs from accepted COW state")
                old = current if present else None
                new = None if after is None else _freeze_dto(after)
                if old is not None and old["buffer_domain_id"] != descriptor.identity.buffer_domain_id:
                    raise PipelineLedgerError(
                        "slot patch crosses the transition buffer domain")
                if new is not None and new["buffer_domain_id"] != descriptor.identity.buffer_domain_id:
                    raise PipelineLedgerError(
                        "slot patch crosses the transition buffer domain")
                leg_data = selected
                endpoint_name = leg_data[3] if leg_data is not None else None
                if old is not None and old["occupied"] and (
                        new is None or not new["occupied"]):
                    old_owner = old["owner"]
                    allowed_clear = (
                        transition["transition_kind"] == "request_close"
                        and descriptors[old_owner["segment_id"]].identity.request_id
                            == descriptor.identity.request_id
                        and descriptors[old_owner["segment_id"]].identity.buffer_domain_id
                            == descriptor.identity.buffer_domain_id)
                    if (transition["transition_kind"] == "consume_done"
                            and old["role"] == "target"
                            and old_owner["segment_id"]
                                == transition["descriptor_segment_id"]):
                        owner_segment = old_owner["segment_id"]
                        owner_descriptor = descriptors.get(owner_segment)
                        owner_shards = tuple(
                            old["target_shards_to_consume"])
                        selected_consumer_proof = selected is None
                        if selected is not None:
                            run, _unused_global_ordinal, shard, endpoint = selected
                            selector = transition["endpoint_selector"]
                            selected_consumer_proof = (
                                endpoint == "body_marker"
                                and run.rank == old["rank"]
                                and run.group_ordinal
                                    == old_owner["group_ordinal"]
                                and selector["tile_ordinal"]
                                    == old_owner["tile_ordinal"]
                                and shard.shard_ordinal in owner_shards
                                and endpoint_completion_proven(
                                    owner_segment, owner_descriptor,
                                    old["rank"], old_owner["group_ordinal"],
                                    old_owner["tile_ordinal"],
                                    shard.shard_ordinal, "body_marker"))
                        ready_and_terminal = bool(owner_shards)
                        ready_tag = _target_owner_tag(old_owner)
                        for target_shard in owner_shards:
                            if not endpoint_completion_proven(
                                    owner_segment, owner_descriptor,
                                    old["rank"], old_owner["group_ordinal"],
                                    old_owner["tile_ordinal"], target_shard,
                                    "data_recv"):
                                ready_and_terminal = False
                                break
                            ready_key = (
                                old["buffer_domain_id"], old["rank"],
                                old["slot_index"], target_shard)
                            already_ready = (
                                target_ready_overlay.get(ready_key)
                                == ready_tag)
                            if not already_ready:
                                # A physical data_recv callback publishes
                                # endpoint_done, consumes terminal watchers,
                                # then publishes data_ready. Its exact cursor
                                # completion may therefore precede this clear
                                # in the same callback; require the matching
                                # semantic data_ready before the delivery ends.
                                pending_ready_after_clear.add(
                                    (ready_key, ready_tag))
                            if not consume_terminal_proven(
                                    owner_descriptor, owner_segment,
                                    old["rank"], old_owner["group_ordinal"],
                                    old_owner["tile_ordinal"], target_shard):
                                ready_and_terminal = False
                                break
                        allowed_clear = (
                            owner_descriptor is not None
                            and bool(owner_shards)
                            and selected_consumer_proof
                            and ready_and_terminal)
                    if (transition["transition_kind"] == "endpoint_done"
                            and endpoint_name == "release_ack_recv"
                            and old["role"] == "source"
                            and leg_data is not None):
                        run, selected_global_tile_ordinal, shard, _endpoint = leg_data
                        allowed_clear = (
                            old["rank"] == run.rank
                            and old["shard_ordinal"] == shard.shard_ordinal)
                        if allowed_clear:
                            release_lag = descriptor.protocol.release_lag_tiles
                            released_tile = (
                                selected_global_tile_ordinal - release_lag)
                            owner_descriptor = descriptors[old_owner["segment_id"]]
                            owner_index = (self._selector_indexes.get(
                                old_owner["segment_id"])
                                or parsed.selector_index_additions.get(
                                    old_owner["segment_id"]))
                            owner_global_tile = _global_tile_ordinal(
                                owner_descriptor, old_owner["group_ordinal"],
                                old_owner["tile_ordinal"], owner_index)
                            owner_run = (None if owner_index is None else
                                _descriptor_run_at(
                                    owner_index, old_owner["group_ordinal"],
                                    old["rank"], old_owner["tile_ordinal"]))
                            allowed_clear = (
                                released_tile > 0
                                and owner_global_tile == released_tile
                                and owner_run is not None
                                and owner_descriptor.identity.request_id
                                    == descriptor.identity.request_id
                                and owner_descriptor.identity.buffer_domain_id
                                    == descriptor.identity.buffer_domain_id
                                and owner_descriptor.identity.generation
                                    == descriptor.identity.generation
                                and owner_descriptor.identity.stage_epoch
                                    == descriptor.identity.stage_epoch
                                and owner_descriptor.identity.staging_allocation_id
                                    == descriptor.identity.staging_allocation_id)
                    if not allowed_clear:
                        raise PipelineLedgerError(
                            "occupied pipeline slot was cleared before its legal release")
                if new is not None and new["occupied"]:
                    owner_segment = new["owner"]["segment_id"]
                    if transition["transition_kind"] == "request_close":
                        raise PipelineLedgerError(
                            "request_close cannot leave an occupied pipeline slot")
                    if transition["transition_kind"] not in (
                            "tile_issue", "consume_start", "fill_ack_progress",
                            "release_ack_progress", "request_close",
                            "consume_done"):
                        raise PipelineLedgerError(
                            "pipeline slot ownership changed outside an issue/consume/close transition")
                    if (transition["transition_kind"] == "request_close"
                            and descriptors[owner_segment].identity.request_id
                                != descriptor.identity.request_id):
                        raise PipelineLedgerError(
                            "request_close touches a foreign request's buffer domain")
                    if (transition["transition_kind"] != "request_close"
                            and owner_segment != transition["descriptor_segment_id"]):
                        raise PipelineLedgerError(
                            "pipeline slot issue assigned a different descriptor owner")
                    if (transition["transition_kind"] in (
                            "request_close", "consume_done")
                            and old is not None and old["occupied"]
                            and old["owner"] != new["owner"]):
                        raise PipelineLedgerError(
                            "administrative close/consume cannot transfer slot ownership")
                    if leg_data is not None and (
                            new["owner"]["tile_ordinal"]
                            != transition["endpoint_selector"]["tile_ordinal"]):
                        raise PipelineLedgerError(
                            "pipeline slot owner tile differs from selected endpoint")
                    if leg_data is not None:
                        run, _unused_global_ordinal, shard, _endpoint = leg_data
                        if new["owner"]["group_ordinal"] != run.group_ordinal:
                            raise PipelineLedgerError(
                                "pipeline slot owner group differs from selected endpoint")
                        if new["role"] == "source" and (
                                new["rank"] != run.rank
                                or new["shard_ordinal"] != shard.shard_ordinal):
                            raise PipelineLedgerError(
                                "source slot owner differs from selected endpoint rank/shard")
                        if new["role"] == "target" and (
                                new["rank"] != shard.target_rank):
                            raise PipelineLedgerError(
                                "target slot owner differs from selected endpoint rank")
                if old is not None and old["occupied"] and new is not None and new["occupied"]:
                    old_owner, new_owner = old["owner"], new["owner"]
                    if old_owner["segment_id"] != new_owner["segment_id"]:
                        old_state = touched_state.get(
                            old_owner["segment_id"],
                            states.get(old_owner["segment_id"]))
                        if (old_state is None or not (
                                old_state["settle_done"] or old_state["retired"])):
                            raise PipelineLedgerError(
                                "foreign slot owner was displaced before settle")
                    else:
                        owner_segment = old_owner["segment_id"]
                        owner_index = (self._selector_indexes.get(owner_segment)
                                       or parsed.selector_index_additions.get(
                                           owner_segment))
                        owner_descriptor = descriptors[owner_segment]
                        old_owner_global_tile = _global_tile_ordinal(
                            owner_descriptor, old_owner["group_ordinal"],
                            old_owner["tile_ordinal"], owner_index)
                        new_owner_global_tile = _global_tile_ordinal(
                            owner_descriptor, new_owner["group_ordinal"],
                            new_owner["tile_ordinal"], owner_index)
                        if old_owner_global_tile != new_owner_global_tile:
                            # In-domain two-slot reuse requires a completed
                            # lagged ReleaseAckRecv watermark for this slot.
                            if (old["role"] != "source" or leg_data is None
                                    or endpoint_name not in (
                                        "data_send", "fill_ack_recv")):
                                raise PipelineLedgerError(
                                    "slot owner changed before a matching release completion")
                            watermark_key = (
                                old["buffer_domain_id"], old["rank"],
                                old["shard_ordinal"], old["slot_index"])
                            if old_owner_global_tile > release_watermarks.get(
                                    watermark_key):
                                raise PipelineLedgerError(
                                    "source slot was reused before its release watermark")
                if old is not None:
                    add_row_aggregate(old, -1)
                if key[1] == "target":
                    old_tag = (None if old is None or not old["occupied"]
                               else _target_owner_tag(old["owner"]))
                    new_tag = (None if new is None or not new["occupied"]
                               else _target_owner_tag(new["owner"]))
                    if (old_tag is not None
                            and (new_tag is None or old_tag != new_tag)):
                        for target_shard in old["target_shards_to_consume"]:
                            target_ready_overlay.delete((
                                key[0], key[2], key[4], target_shard))
                if new is not None:
                    slot_overlay.set(key, new)
                    add_row_aggregate(new, 1)
                else:
                    slot_overlay.delete(key)
                if key[1] == "source":
                    touched_keys.add((key[0], key[2], key[3]))
                else:
                    touched_targets.add((key[0], key[2]))
            for delta in target_deltas:
                touched_targets.update(apply_target_delta(
                    transition, descriptor, selected, delta))
            return touched_keys, touched_targets

        def apply_credit_patches(transition, expected_ack):
            touched = set()
            for key, before, after in transition["credit_patches"]:
                present = credit_overlay.has(key)
                current = credit_overlay.get(key)
                if before is None:
                    if present:
                        raise PipelineLedgerError(
                            "credit patch expected an absent before row")
                elif not present or not _typed_equal(current, before):
                    raise PipelineLedgerError(
                        "credit patch before row differs from accepted COW state")
                old = current if present else None
                new = None if after is None else _freeze_dto(after)
                if new is not None and _credit_key(new) != key:
                    raise PipelineLedgerError("credit patch changes its stable key")
                domain = key[0]
                if domain != descriptor.identity.buffer_domain_id:
                    raise PipelineLedgerError(
                        "credit patch crosses the transition buffer domain")
                old_ack = 0 if old is None else old["outstanding_ack_count"]
                new_ack = 0 if new is None else new["outstanding_ack_count"]
                expected_delta = expected_ack[1] if expected_ack[0] == key else 0
                if new_ack != old_ack + expected_delta:
                    raise PipelineLedgerError(
                        "credit ACK count differs from its sequenced send/completion")
                if (old is not None and new is None
                        and (old_ack != 0 or source_occupancy.get(key) != 0)):
                    raise PipelineLedgerError(
                        "credit row tombstone retains occupied slots or ACKs")
                if old is not None:
                    credit_ack_domains.add(domain, -old["outstanding_ack_count"])
                    credit_row_counts.add(domain, -1)
                if new is not None:
                    credit_overlay.set(key, new)
                    credit_ack_domains.add(domain, new["outstanding_ack_count"])
                    credit_row_counts.add(domain, 1)
                else:
                    credit_overlay.delete(key)
                touched.add(key)
            if expected_ack[1] and expected_ack[0] not in touched:
                raise PipelineLedgerError(
                    "release ACK count changed without a matching credit row patch")
            return touched

        def apply_obligation_patches(transition, required_expected):
            seen = set()
            postwrite_verified = None
            for key, before, after in transition["obligation_patches"]:
                present = obligation_overlay.has(key)
                current = obligation_overlay.get(key)
                if before is None:
                    if present:
                        raise PipelineLedgerError(
                            "obligation patch expected an absent before row")
                elif not present or not _typed_equal(current, before):
                    raise PipelineLedgerError(
                        "obligation patch before row differs from accepted COW state")
                old = current if present else None
                new = None if after is None else _freeze_dto(after)
                if new is not None and new["obligation_id"] != key:
                    raise PipelineLedgerError(
                        "obligation patch changes its stable obligation_id")
                if old is not None:
                    obligation_remaining.add(
                        old["segment_id"],
                        -old["remaining_endpoint_count"])
                if new is not None:
                    obligation_overlay.set(key, new)
                    obligation_remaining.add(
                        new["segment_id"], new["remaining_endpoint_count"])
                else:
                    obligation_overlay.delete(key)
                seen.add(key)
                expected = required_expected.get(key, _ROW_TOMBSTONE)
                expected_row = None if expected is _ROW_TOMBSTONE else expected
                if not _typed_equal(new, expected_row):
                    raise PipelineLedgerError(
                        "obligation patch does not match endpoint completion/close progress")
                # _RowOverlay stores ``new`` by identity. For parser-owned
                # builtin rows the comparison above therefore also proves the
                # post-write value. Keep the historical second comparison for
                # custom strings/mappings (including subtype equality side
                # effects) retained by the old or new row.
                if (type(key) is str and key in required_expected
                        and _plain_obligation_row_for_postwrite(old)
                        and _plain_obligation_row_for_postwrite(new)):
                    if postwrite_verified is None:
                        postwrite_verified = set()
                    postwrite_verified.add(key)
            if seen != set(required_expected):
                raise PipelineLedgerError(
                    "obligation patch set does not match endpoint progress")
            # Keep the original required_expected iteration/error order for
            # every row whose equality can dispatch to user code.
            for key, expected in required_expected.items():
                if (postwrite_verified is not None and type(key) is str
                        and key in postwrite_verified):
                    continue
                current = obligation_overlay.get(key)
                expected_row = None if expected is _ROW_TOMBSTONE else expected
                if not _typed_equal(current, expected_row):
                    raise PipelineLedgerError(
                        "obligation state is inconsistent with cursor progress")

        def verify_credit_balance(keys, target_keys):
            for key in keys:
                occupancy = source_occupancy.get(key)
                credit = credit_overlay.get(key)
                if occupancy < 0 or occupancy > 2:
                    raise PipelineLedgerError(
                        "source pipeline slot occupancy exceeds two-slot capacity")
                if credit is None:
                    if occupancy:
                        raise PipelineLedgerError(
                            "occupied source slots have no matching credit row")
                    continue
                if (credit["capacity"] != 2
                        or credit["available"] != credit["capacity"] - occupancy):
                    raise PipelineLedgerError(
                        "source credit balance does not match occupied slots")
                if credit["outstanding_ack_count"] < 0:
                    raise PipelineLedgerError("source outstanding ACK count underflow")
            for key in target_keys:
                if target_occupancy.get(key) < 0 \
                        or target_occupancy.get(key) > 2:
                    raise PipelineLedgerError(
                        "target pipeline slot occupancy exceeds two-slot capacity")

        def simulate_issue_or_completion(
                run_transition, descriptor, progress, segment, run_index):
            transition = run_transition["transition_kind"]
            store_id = run_transition["endpoint_store_id"]
            ordinal = run_transition["endpoint_ordinal"]
            if transition == "endpoint_done" or (
                    transition == "tile_issue" and store_id in progress["active"]):
                if store_id not in progress["active"]:
                    raise PipelineLedgerError(
                        "endpoint completion has no active issue")
                active_ordinal, _active_row = progress["active"].pop(store_id)
                if active_ordinal != ordinal:
                    raise PipelineLedgerError(
                        "endpoint completion selector differs from active issue")
                merge_completion(
                    progress, ordinal, "endpoint_done", segment, run_index)
                return "complete"
            if transition not in (
                    "tile_issue", "consume_start", "fill_ack_progress",
                    "release_ack_progress"):
                return None
            if progress["closed"]:
                raise PipelineLedgerError(
                    "request_close forbids new endpoint issue")
            if ordinal != progress["position"]:
                raise PipelineLedgerError(
                    "endpoint issue does not match the next legal cursor position")
            if store_id in progress["active"]:
                raise PipelineLedgerError("endpoint issue duplicates an active store ID")
            progress["active"][store_id] = (ordinal, None)
            progress["position"] += 1
            if progress["position"] > progress["total"]:
                raise PipelineLedgerError("endpoint cursor advanced beyond its run")
            cursor_dirty.add((segment, run_index))
            cursor_unissued.add(segment, -1)
            advance_inactive_positions(
                descriptor, progress, segment, run_index, run_transition)
            return "issue"

        for transition in pipeline_rows:
            sequence = transition["first_logical_seq"]
            stage_node_events_before(sequence)
            segment = transition["descriptor_segment_id"]
            descriptor = descriptors.get(segment)
            if descriptor is None:
                raise PipelineLedgerError("pipeline transition references unknown descriptor")
            state = current_state(segment)
            kind = transition["transition_kind"]
            domain = descriptor.identity.buffer_domain_id
            descriptor_transition_counts[segment] = (
                descriptor_transition_counts.get(segment, 0) + 1)
            if state["retired"]:
                raise PipelineLedgerError(
                    "retired descriptor received a later transition")
            if kind == "descriptor_commit":
                if segment not in additions or segment in descriptor_commits:
                    raise PipelineLedgerError(
                        "descriptor_commit must declare exactly one new descriptor")
                descriptor_commits.add(segment)
                if transition["cursor_version_start"] != state["cursor_version"]:
                    raise PipelineLedgerError(
                        "descriptor_commit cursor pre-version disagrees with declaration")
                entry = initial_state_updates[segment]
                if state["service_done"] != entry["service_done"] \
                        or state["settle_done"] != entry["settle_done"] \
                        or state["retired"] != entry["retired"]:
                    raise PipelineLedgerError(
                        "descriptor_commit metadata disagrees with declaration")
                active_domain_counts.add(domain, 1)
                for initial_run_index in range(len(descriptor.endpoint_runs)):
                    initial_progress = seed_cursor_progress(
                        segment, initial_run_index)
                    cursor_unissued.add(
                        segment, initial_progress["total"]
                        - initial_progress["position"])
                    initial_obligation = expected_obligation(
                        descriptor, segment, initial_run_index,
                        initial_progress)
                    obligation_id = f"{segment}/run/{initial_run_index}"
                    expected_obligation_rows[obligation_id] = (
                        _ROW_TOMBSTONE if initial_obligation is None
                        else initial_obligation)
                    active_endpoint_deltas[segment] = (
                        active_endpoint_deltas.get(segment, 0)
                        + len(initial_progress["active"]))
            elif (segment not in self._active_segments
                  and segment not in descriptor_commits):
                raise PipelineLedgerError(
                    "pipeline transition precedes descriptor_commit")
            current_version = cursor_versions.get(
                segment, state["cursor_version"])
            if transition["cursor_version_start"] != current_version:
                raise PipelineLedgerError(
                    "pipeline transition cursor pre-version mismatch")
            next_version = transition["cursor_version_end"]
            cursor_versions[segment] = next_version
            state["cursor_version"] = next_version

            selected = selected_slot_and_credit(descriptor, transition)
            run_index = transition["endpoint_run_index"]
            progress_key = ((segment, run_index)
                            if run_index is not None else None)
            if progress_key is not None:
                progress = seed_cursor_progress(segment, run_index)
                cursor_dirty.add(progress_key)
            else:
                progress = None

            progress_action = None
            if kind == "request_close":
                if state.get("request_closed", False):
                    raise PipelineLedgerError("request_close transition repeated")
                for index in range(len(descriptor.endpoint_runs)):
                    run_progress = seed_cursor_progress(segment, index)
                    if run_progress["closed"]:
                        raise PipelineLedgerError(
                            "request_close found a previously closed cursor run")
                    completed = _completed_count(run_progress)
                    unresolved = (run_progress["total"] - completed
                                  - run_progress["cancelled"])
                    active_count = len(run_progress["active"])
                    if unresolved < active_count:
                        raise PipelineLedgerError(
                            "request_close active endpoints exceed unresolved work")
                    cursor_unissued.add(
                        segment, -(run_progress["total"]
                                   - run_progress["position"]))
                    run_progress["cancelled"] += unresolved - active_count
                    run_progress["remaining"] = active_count
                    run_progress["position"] = run_progress["total"]
                    run_progress["closed"] = True
                    cursor_dirty.add((segment, index))
                    old = expected_obligation(
                        descriptor, segment, index, run_progress)
                    key = f"{segment}/run/{index}"
                    expected_obligation_rows[key] = (
                        _ROW_TOMBSTONE if old is None else old)
                state["request_closed"] = True
                state["service_done"] = True
            elif progress is not None:
                progress_action = simulate_issue_or_completion(
                    transition, descriptor, progress, segment, run_index)
                if progress_action is not None:
                    new_expected = expected_obligation(
                        descriptor, segment, run_index, progress)
                    key = f"{segment}/run/{run_index}"
                    expected_obligation_rows[key] = (
                        _ROW_TOMBSTONE if new_expected is None else new_expected)
                    if progress_action == "complete":
                        # endpoint_done carries the concrete descriptor/run/
                        # selector and affine store id. It is terminal proof
                        # for this exact logical endpoint; request_close
                        # cancellation and cursor prefix normalization never
                        # enter this path.
                        completed_run = descriptor.endpoint_runs[run_index]
                        _record_dependency_terminal(
                            completed_run.rank,
                            transition["endpoint_store_id"], {segment})
            if kind == "data_ready":
                if selected is None or selected[3] != "data_recv":
                    raise PipelineLedgerError(
                        "data_ready must bind a concrete data_recv endpoint")
                _run, global_tile_ordinal, shard, _endpoint = selected
                ordinal = transition["endpoint_ordinal"]
                if progress is None or not progress_has_completed(
                        progress, ordinal):
                    raise PipelineLedgerError(
                        "data_ready precedes data_recv endpoint completion")
                tile_ordinal = transition["endpoint_selector"]["tile_ordinal"]
                slot_index = (global_tile_ordinal - 1) % 2
                slot_key = (
                    domain, "target", _run.rank, None,
                    slot_index)
                ready_slot_key = (
                    domain, _run.rank, slot_index,
                    shard.shard_ordinal)
                target_row = current_target_row(slot_key)
                owner = None if target_row is None else target_row["owner"]
                pending_key = (ready_slot_key, (
                    segment, descriptor.identity.generation,
                    descriptor.identity.stage_epoch,
                    descriptor.identity.staging_allocation_id,
                    _run.group_ordinal, tile_ordinal))
                if (state.get("request_closed", False)
                        and (target_row is None
                             or not target_row["occupied"])):
                    if target_ready_overlay.get(ready_slot_key) is not None:
                        raise PipelineLedgerError(
                            "closed descriptor retains a ready target slot")
                    pending_ready_after_clear.discard(pending_key)
                elif (target_row is not None and target_row["occupied"]
                        and target_row["shard_ordinal"] is None
                        and _sorted_uint_tuple_contains(
                            target_row["target_shards_to_consume"],
                            shard.shard_ordinal)
                        and owner is not None
                        and owner["segment_id"] == segment
                        and owner["group_ordinal"] == _run.group_ordinal
                        and owner["tile_ordinal"] == tile_ordinal
                        and owner["pipeline_generation"]
                            == descriptor.identity.generation
                        and owner["stage_epoch"] == descriptor.identity.stage_epoch
                        and owner["staging_allocation_id"]
                            == descriptor.identity.staging_allocation_id):
                    tag = _target_owner_tag(owner)
                    ready_owner = target_ready_overlay.get(ready_slot_key)
                    if ready_owner is not None and ready_owner != tag:
                        raise PipelineLedgerError(
                            "ready target slot belongs to a different owner")
                    if ready_owner == tag:
                        raise PipelineLedgerError(
                            "target shard received duplicate data_ready")
                    target_ready_overlay.set(ready_slot_key, tag)
                elif pending_key in pending_ready_after_clear:
                    pending_ready_after_clear.remove(pending_key)
                else:
                    expected_owner = (
                        segment, descriptor.identity.generation,
                        descriptor.identity.stage_epoch,
                        descriptor.identity.staging_allocation_id,
                        _run.group_ordinal, tile_ordinal)
                    raise PipelineLedgerError(
                        "data_ready owner mismatch at seq={} selector={!r} "
                        "slot_key={!r} expected_owner={!r} actual_owner={!r}"
                        .format(sequence, transition["endpoint_selector"],
                                slot_key, expected_owner, owner))
            elif kind == "service_done":
                if state["service_done"]:
                    raise PipelineLedgerError("service_done transition repeated")
                # Service completion closes issue production; submitted legs
                # still drain until settle_done.
                if not state.get("request_closed") \
                        and cursor_unissued.get(segment) != 0:
                    raise PipelineLedgerError(
                        "service_done precedes cursor issue exhaustion")
                state["service_done"] = True
            elif kind == "settle_done":
                if state["settle_done"]:
                    raise PipelineLedgerError("settle_done transition repeated")
                if not state["service_done"]:
                    raise PipelineLedgerError(
                        "settle_done precedes service_done/request_close")
                if obligation_remaining.get(segment) != 0:
                    raise PipelineLedgerError(
                        "settle_done retains endpoint obligations")
                if self._active_endpoint_counts.get(segment, 0) \
                        + active_endpoint_deltas.get(segment, 0) != 0:
                    raise PipelineLedgerError(
                        "settle_done retains active physical endpoints")
                state["settle_done"] = True
            elif kind == "descriptor_retire":
                if not state["service_done"] or not state["settle_done"]:
                    raise PipelineLedgerError(
                        "descriptor retirement precedes service and settle")
                if obligation_remaining.get(segment) != 0:
                    raise PipelineLedgerError(
                        "descriptor retirement retains endpoint obligations")
                if slot_owner_counts.get(segment) != 0:
                    raise PipelineLedgerError(
                        "descriptor retirement retains a slot owner")
                if ack_pins_by_segment.get(segment) != 0:
                    raise PipelineLedgerError(
                        "descriptor retirement retains an owned ACK pin")
                if self._active_endpoint_counts.get(segment, 0) \
                        + active_endpoint_deltas.get(segment, 0) != 0:
                    raise PipelineLedgerError(
                        "descriptor retirement retains an active endpoint owner")
                if resource_counts.get(segment, 0):
                    raise PipelineLedgerError(
                        "descriptor retirement retains an active resource")
                if dependency_pins.get(segment) != 0:
                    raise PipelineLedgerError(
                        "descriptor retirement retains an unfinished ordinary dependency pin")
                active_domain_counts.add(domain, -1)
                if active_domain_counts.get(domain) < 0:
                    raise PipelineLedgerError(
                        "active descriptor buffer-domain count underflow")
                if active_domain_counts.get(domain) == 0:
                    domain_retirements.add(domain)
                state["retired"] = True
                retirements.append(segment)
            elif kind == "descriptor_commit":
                pass

            # Release ACK pins change at the actual completion callback, which
            # precedes endpoint_done in the producer's logical stream.
            ack_key = None
            ack_delta = 0
            ack_noop_closed_recv = False
            if selected is not None:
                run, _unused_global_ordinal, shard, endpoint = selected
                key = (domain, shard.source_rank, shard.shard_ordinal)
                if kind == "release_ack_progress" \
                        and endpoint == "release_ack_send":
                    ack_key, ack_delta = key, 1
                elif kind == "endpoint_done" \
                        and endpoint == "release_ack_recv":
                    row = credit_overlay.get(key)
                    current_ack = 0 if row is None else row["outstanding_ack_count"]
                    if current_ack == 0:
                        if not state.get("request_closed", False):
                            raise PipelineLedgerError(
                                "release ACK receive lacks a pending source ACK pin")
                        ack_noop_closed_recv = True
                    else:
                        ack_key, ack_delta = key, -1
            source_keys, target_keys = apply_slot_patches(
                transition, descriptor, selected)
            credit_keys = apply_credit_patches(
                transition, (ack_key, ack_delta))
            for key in source_keys:
                credit_keys.add(key)
            if ack_delta:
                credit_keys.add(ack_key)
                ack_pins_by_segment.add(segment, ack_delta)
                if ack_pins_by_segment.get(segment) < 0:
                    raise PipelineLedgerError("segment release ACK pin underflow")
            verify_credit_balance(credit_keys, target_keys)
            if ack_noop_closed_recv and ack_key is not None:
                raise PipelineLedgerError(
                    "closed no-op ACK receive unexpectedly changed a credit row")

            if selected is not None:
                run, global_tile_ordinal, shard, endpoint = selected
                if kind == "endpoint_done" and endpoint == "release_ack_recv":
                    lag = descriptor.protocol.release_lag_tiles
                    if global_tile_ordinal > lag:
                        released_tile = global_tile_ordinal - lag
                        slot_index = (released_tile - 1) % 2
                        watermark_key = (
                            domain, run.rank, shard.shard_ordinal, slot_index)
                        if released_tile > release_watermarks.get(watermark_key):
                            release_watermarks.add(
                                watermark_key,
                                released_tile - release_watermarks.get(watermark_key))
                if (old := transition["slot_patches"]):
                    del old  # touched-slot legality was checked by its COW applier

            if kind == "descriptor_retire" and domain in domain_retirements:
                if (slot_row_counts.get(domain) != 0
                        or credit_row_counts.get(domain) != 0
                        or credit_ack_domains.get(domain) != 0):
                    raise PipelineLedgerError(
                        "last descriptor retirement retains buffer-domain rows or ACK pins")

            required_obligations = dict(expected_obligation_rows)
            # Only the obligation keys whose semantic row changes belong in
            # this transition. Unchanged endpoint issues produce no wire row.
            changed_expected = {}
            for key, expected in required_obligations.items():
                current = obligation_overlay.get(key)
                expected_row = None if expected is _ROW_TOMBSTONE else expected
                if not _typed_equal(current, expected_row):
                    changed_expected[key] = expected
            # A transition with no obligation effect must carry no obligation
            # patch; all other wire changes must exactly equal derived progress.
            apply_obligation_patches(transition, changed_expected)
            expected_obligation_rows.clear()

            if selected is not None and progress_key is not None:
                if progress_action == "issue":
                    active_endpoint_deltas[segment] = (
                        active_endpoint_deltas.get(segment, 0) + 1)
                elif progress_action == "complete":
                    active_endpoint_deltas[segment] = (
                        active_endpoint_deltas.get(segment, 0) - 1)
            if kind == "request_close":
                # Close does not erase issued endpoints; its active count is
                # unchanged. Only unissued tail work is canceled.
                pass

        stage_node_events_before(parsed.seq_end + 1)
        if pending_ready_after_clear:
            raise PipelineLedgerError(
                "target consume callback lacks its sequenced data_ready marker")
        if work_transaction is not None:
            work_transaction.finish(parsed.seq_end)

        if descriptor_commits != set(additions):
            raise PipelineLedgerError(
                "every descriptor declaration needs exactly one descriptor_commit transition")
        retired_segment_set = set(retirements)
        required_cursor_segments = set(additions) | set(retirements)
        required_cursor_segments.update(segment for segment, _ in cursor_dirty)
        for segment, version in cursor_versions.items():
            initial = initial_state_updates.get(segment)
            base_version = (initial["cursor_version"] if initial is not None
                            else self._cursor_versions_by_segment.get(segment))
            if base_version != version:
                required_cursor_segments.add(segment)
        if required_cursor_segments != set(parsed_cursor_patches):
            raise PipelineLedgerError(
                "cursor patches do not exactly cover initialized, retired, or changed cursors")
        for segment in retirements:
            patch = parsed_cursor_patches.get(segment)
            if patch is None or not patch["retired"]:
                raise PipelineLedgerError(
                    "descriptor retirement needs explicit cursor run tombstones")
        for segment in additions:
            patch = parsed_cursor_patches.get(segment)
            descriptor = additions[segment]
            if patch is None:
                raise PipelineLedgerError(
                    "new descriptor requires initial cursor run rows")
            expected_indexes = list(range(len(descriptor.endpoint_runs)))
            if [index for index, _row in patch["runs"]] != expected_indexes:
                raise PipelineLedgerError(
                    "new descriptor cursor patch must initialize every run")
            if patch["retired"]:
                raise PipelineLedgerError(
                    "new descriptor cannot start retired")

        for patch in parsed.cursor_patches:
            segment = patch["segment_id"]
            descriptor = descriptors[segment]
            state = current_state(segment)
            if descriptor_transition_counts.get(segment, 0) == 0:
                raise PipelineLedgerError(
                    "cursor patch has no sequenced descriptor transition")
            if patch["cursor_version"] != cursor_versions.get(
                    segment, state["cursor_version"]):
                raise PipelineLedgerError(
                    "final cursor patch version disagrees with transition sequence")
            if patch["retired"] != state["retired"]:
                raise PipelineLedgerError(
                    "cursor retirement marker disagrees with descriptor lifecycle")
            cursor_version_updates[segment] = patch["cursor_version"]
            base_cursor_version = (
                initial_state_updates[segment]["cursor_version"]
                if segment in initial_state_updates else
                self._cursor_versions_by_segment.get(segment))
            if (not patch["runs"] and segment not in additions
                    and patch["cursor_version"] == base_cursor_version):
                raise PipelineLedgerError(
                    "metadata-only cursor patch has no sequenced version change")
            if patch["retired"]:
                if segment not in retired_segment_set:
                    raise PipelineLedgerError(
                        "cursor tombstones require descriptor_retire transition")
                cursor_changes.update({
                    (segment, run_index): _ROW_TOMBSTONE
                    for run_index, _value in patch["runs"]})
                cursor_transfer_changes.update({
                    (segment, run_index): _ROW_TOMBSTONE
                    for run_index, _value in patch["runs"]})
                for run_index, _value in patch["runs"]:
                    old_row = (self._cursor_runs_by_segment.get(
                        segment, {}).get(run_index)
                        if segment not in additions else
                        _initial_cursor_run(descriptor, segment, run_index))
                    if old_row is None:
                        raise PipelineLedgerError(
                            "retirement tombstones a missing accepted cursor run")
                    old_active = old_row["active_endpoints"]
                    if old_active:
                        cursor_active_deltas[segment] = (
                            cursor_active_deltas.get(segment, 0)
                            - len(old_active))
                    for endpoint in old_active:
                        transfer_id = endpoint["transfer_id"]
                        active_transfer_id_counts.add(transfer_id, -1)
                        if active_transfer_id_counts.get(transfer_id) < 0:
                            raise PipelineLedgerError(
                                "active transfer index underflow during retirement")
                continue
            for run_index, final_row in patch["runs"]:
                key = (segment, run_index)
                progress = seed_cursor_progress(segment, run_index)
                if key not in cursor_dirty and segment not in additions:
                    raise PipelineLedgerError(
                        "cursor run patch has no matching issue/completion/close transition")
                if final_row is None:
                    raise PipelineLedgerError(
                        "nonretired cursor patch cannot tombstone a run")
                current_row = (None if segment in additions else
                    self._cursor_runs_by_segment.get(segment, {}).get(run_index))
                if current_row is not None and _typed_equal(current_row, final_row):
                    raise PipelineLedgerError(
                        "cursor run patch redundantly republishes an unchanged row")
                total = progress["total"]
                completed_count = _completed_count(progress)
                expected_remaining = (len(progress["active"])
                    if progress["closed"] else total - completed_count
                    - progress["cancelled"])
                if (final_row["completed_endpoint_count"] != progress["completed"]
                        or final_row["cancelled_endpoint_count"] != progress["cancelled"]
                        or final_row["remaining_endpoint_count"] != expected_remaining
                        or final_row["in_flight"] != bool(progress["active"])):
                    raise PipelineLedgerError(
                        "cursor run counts disagree with sequenced endpoint progress")
                final_ooo = tuple(
                    (item["first_ordinal"], item["end_ordinal_exclusive"])
                    for item in final_row["completed_out_of_order"])
                simulated_ooo = (
                    _completed_interval_export_half_open(progress["ooo_root"])
                    if progress["ooo_changed"] else
                    progress["ooo_wire_ranges"])
                if final_ooo != simulated_ooo:
                    raise PipelineLedgerError(
                        "cursor out-of-order ranges disagree with endpoint completions")
                if _cursor_position_ordinal(final_row, progress["run"]) != progress["position"]:
                    raise PipelineLedgerError(
                        "cursor next position disagrees with sequenced endpoint issues")
                if final_row["next_store_id"] is not None:
                    if final_row["active_endpoints"] is None:
                        raise PipelineLedgerError("cursor has an invalid active endpoint list")
                final_active = {
                    item["store_id"]: item
                    for item in final_row["active_endpoints"]
                }
                if set(final_active) != set(progress["active"]):
                    raise PipelineLedgerError(
                        "cursor active endpoint set disagrees with sequenced issue/completion events")
                old_active_rows = {}
                if current_row is not None:
                    old_active_rows = {
                        item["store_id"]: item
                        for item in current_row["active_endpoints"]
                    }
                for store_id, active in final_active.items():
                    old = old_active_rows.get(store_id)
                    if old is None:
                        continue
                    if (old["transfer_id"] != active["transfer_id"]
                            or old["owner"] != active["owner"]
                            or old["tile_ordinal"] != active["tile_ordinal"]
                            or old["shard_ordinal"] != active["shard_ordinal"]
                            or old["endpoint"] != active["endpoint"]
                            or (old["flow_id"] is not None
                                and active["flow_id"] != old["flow_id"])
                            or (old["network_done"] and not active["network_done"])
                            or (old["hbm_done"] and not active["hbm_done"])):
                        raise PipelineLedgerError(
                            "active physical endpoint identity/progress regressed")
                cursor_changes[key] = _freeze_dto(final_row)
                cursor_transfer_changes[key] = MappingProxyType(
                    {store_id: _freeze_dto(item)
                     for store_id, item in final_active.items()})
                delta = len(final_active) - len(old_active_rows)
                if delta:
                    cursor_active_deltas[segment] = (
                        cursor_active_deltas.get(segment, 0) + delta)
                for endpoint in old_active_rows.values():
                    active_transfer_id_counts.add(
                        endpoint["transfer_id"], -1)
                for endpoint in final_active.values():
                    transfer_id = endpoint["transfer_id"]
                    active_transfer_id_counts.add(transfer_id, 1)
                    if active_transfer_id_counts.get(transfer_id) > 1:
                        raise PipelineLedgerError(
                            "active transfer_id is already owned by another cursor")

        for segment, delta in active_endpoint_deltas.items():
            if delta != cursor_active_deltas.get(segment, 0):
                raise PipelineLedgerError(
                    "active endpoint aggregate differs from final cursor rows")
        for segment, delta in cursor_active_deltas.items():
            if delta != active_endpoint_deltas.get(segment, 0):
                raise PipelineLedgerError(
                    "final cursor rows contain an unsequenced active endpoint change")

        # Every run touched by a sequenced issue/completion/close must have a
        # corresponding final full-row patch or retirement tombstone.
        # Declaration initialization is also complete and checked against the
        # descriptor-only normalized prefix; it may then be advanced by
        # transitions in this same batch.
        for key in cursor_dirty:
            patch = parsed_cursor_patches.get(key[0])
            if (patch is None
                    or key[1] not in parsed_cursor_patch_run_indexes.get(
                        key[0], ())):
                raise PipelineLedgerError(
                    "cursor transition is missing its touched run patch")
        # Rows/resources are detached before they become persistent state.
        for resource in parsed.resources:
            if resource["as_of_tick"] != parsed.tick:
                raise PipelineLedgerError(
                    "active resource frame must match envelope tick")
        frozen_resources = tuple(_freeze_dto(row) for row in parsed.resources)
        for (domain, rank), _delta in target_occupancy.deltas.items():
            if target_occupancy.get((domain, rank)) < 0 \
                    or target_occupancy.get((domain, rank)) > 2:
                raise PipelineLedgerError(
                    "target pipeline slot occupancy exceeds two-slot capacity")
        for segment, expected_cursor_version in cursor_versions.items():
            state = touched_state.get(segment, states.get(segment))
            if state is not None and state["cursor_version"] != expected_cursor_version:
                raise PipelineLedgerError(
                    "descriptor state cursor version disagrees with pipeline sequence")

        slot_changes = slot_overlay.changes()
        if parsed.protocol_version == 4:
            for key, workset in target_worksets.items():
                row = workset["row"]
                if row is None:
                    slot_changes[key] = _ROW_TOMBSTONE
                    continue
                frozen_row = dict(row)
                consumed = _freeze_consumed_slot_members(
                    workset["consumed"])
                frozen_row["consumed"] = bool(consumed)
                frozen_row["consumed_target_shards"] = consumed
                slot_changes[key] = _freeze_dto(frozen_row)
        target_ready_changes = target_ready_overlay.changes()
        credit_changes = credit_overlay.changes()
        obligation_changes = obligation_overlay.changes()
        aggregate_deltas = {
            "slot_owner": MappingProxyType(dict(slot_owner_counts.deltas)),
            "source_occupancy": MappingProxyType(dict(source_occupancy.deltas)),
            "target_occupancy": MappingProxyType(dict(target_occupancy.deltas)),
            "credit_ack_domain": MappingProxyType(dict(credit_ack_domains.deltas)),
            "ack_pins_by_segment": MappingProxyType(
                dict(ack_pins_by_segment.deltas)),
            "active_descriptor_by_domain": MappingProxyType(
                dict(active_domain_counts.deltas)),
            "slot_rows_by_domain": MappingProxyType(
                dict(slot_row_counts.deltas)),
            "credit_rows_by_domain": MappingProxyType(
                dict(credit_row_counts.deltas)),
            "obligation_remaining": MappingProxyType(
                dict(obligation_remaining.deltas)),
            "dependency_pins": MappingProxyType(dict(dependency_pins.deltas)),
            "dependency_waiters": MappingProxyType(
                dict(dependency_waiters.deltas)),
            "cursor_unissued": MappingProxyType(dict(cursor_unissued.deltas)),
            "active_transfer_ids": MappingProxyType(
                dict(active_transfer_id_counts.deltas)),
            "source_release_watermarks": MappingProxyType(
                dict(release_watermarks.deltas)),
        }
        visible_change = bool(
            parsed.event_count or parsed.descriptors
            or slot_changes or target_ready_changes or credit_changes
            or obligation_changes
            or cursor_changes or cursor_version_updates
            or not _typed_equal(self._active_resources, frozen_resources))
        expected_version = parsed.base_version + (1 if visible_change else 0)
        if parsed.result_version != expected_version:
            raise PipelineLedgerError(
                "result_state_version does not match visible state change")

        state_updates = {}
        for segment, state in touched_state.items():
            state_updates[segment] = _freeze_dto(state)
        for segment, entry in initial_state_updates.items():
            if segment not in state_updates:
                state_updates[segment] = _freeze_dto(entry)
        return PreparedPipelineUpdate(
            parsed=parsed,
            descriptor_additions=MappingProxyType(dict(additions)),
            descriptor_state_updates=MappingProxyType(state_updates),
            descriptor_retirements=tuple(retirements),
            cursor_changes=MappingProxyType(cursor_changes),
            cursor_version_updates=MappingProxyType(cursor_version_updates),
            cursor_transfer_changes=MappingProxyType(cursor_transfer_changes),
            active_endpoint_deltas=MappingProxyType(active_endpoint_deltas),
            resources=frozen_resources,
            slot_changes=MappingProxyType(slot_changes),
            target_slot_lineage_changes=MappingProxyType(
                dict(target_slot_lineage_changes)),
            target_ready_changes=MappingProxyType(target_ready_changes),
            terminal_provenance_updates=MappingProxyType(
                dict(terminal_provenance_roots)),
            credit_changes=MappingProxyType(credit_changes),
            obligation_changes=MappingProxyType(obligation_changes),
            aggregate_deltas=MappingProxyType(aggregate_deltas),
            selector_index_additions=parsed.selector_index_additions,
            dependency_index_additions=MappingProxyType(
                dict(dependency_root_changes)),
            dependency_rank_index_additions=MappingProxyType(
                dict(dependency_rank_root_changes)),
            changed=visible_change,
        )

    def commit_prepared(self, update: PreparedPipelineUpdate) -> PipelineLedgerReceipt:
        """Publish a previously fully validated update; contains no parsing."""
        parsed = update.parsed
        if parsed.protocol_version != self.PROTOCOL_VERSION:
            raise PipelineLedgerError(
                "prepared pipeline update protocol does not match this ledger")
        if update.replayed:
            self.replay_count += 1
            return PipelineLedgerReceipt(parsed.snapshot_kind,
                parsed.delivery_sequence, parsed.event_count, parsed.base_watermark,
                self.watermark, parsed.base_version, self.state_version, True)

        def apply_count_deltas(target, deltas):
            for key, delta in deltas.items():
                value = target.get(key, 0) + delta
                if value:
                    target[key] = value
                else:
                    target.pop(key, None)

        def apply_row_changes(target, changes):
            for key, value in changes.items():
                if value is _ROW_TOMBSTONE:
                    target.pop(key, None)
                else:
                    target[key] = value

        self._descriptors.update(update.descriptor_additions)
        self._descriptor_state.update(update.descriptor_state_updates)
        self._selector_indexes.update(update.selector_index_additions)
        self._dependency_endpoint_index.update(
            update.dependency_index_additions)
        self._dependency_endpoint_index_by_rank.update(
            update.dependency_rank_index_additions)
        self._completed_terminal_intervals_by_rank.update(
            update.terminal_provenance_updates)
        self._active_segments.update(update.descriptor_additions)
        for segment in update.descriptor_retirements:
            self._active_segments.discard(segment)

        for (segment, run_index), value in update.cursor_changes.items():
            if value is _ROW_TOMBSTONE:
                rows = self._cursor_runs_by_segment.get(segment)
                if rows is not None:
                    rows.pop(run_index, None)
            else:
                self._cursor_runs_by_segment.setdefault(segment, {})[
                    run_index] = value
        for segment, version in update.cursor_version_updates.items():
            self._cursor_versions_by_segment[segment] = version
        for key, value in update.cursor_transfer_changes.items():
            if value is _ROW_TOMBSTONE:
                self._active_transfers_by_run.pop(key, None)
            else:
                self._active_transfers_by_run[key] = value

        aggregate_targets = {
            "slot_owner": self._slot_owner_counts,
            "source_occupancy": self._source_occupancy,
            "target_occupancy": self._target_occupancy,
            "credit_ack_domain": self._credit_ack_counts,
            "ack_pins_by_segment": self._ack_pins_by_segment,
            "active_descriptor_by_domain": self._active_descriptor_count_by_domain,
            "slot_rows_by_domain": self._slot_row_counts_by_domain,
            "credit_rows_by_domain": self._credit_row_counts_by_domain,
            "obligation_remaining": self._obligation_remaining_by_segment,
            "dependency_pins": self._unfinished_dependency_pins_by_segment,
            "dependency_waiters":
                self._unfinished_dependency_waiters_by_endpoint,
            "cursor_unissued": self._cursor_unissued_by_segment,
            "active_transfer_ids": self._active_transfer_id_counts,
            "source_release_watermarks": self._source_release_watermarks,
        }
        for name, target in aggregate_targets.items():
            apply_count_deltas(target, update.aggregate_deltas.get(name, {}))
        for transfer_id, count in update.aggregate_deltas.get(
                "active_transfer_ids", {}).items():
            if count:
                if self._active_transfer_id_counts.get(transfer_id, 0) > 0:
                    self._active_transfer_ids.add(transfer_id)
                else:
                    self._active_transfer_ids.discard(transfer_id)

        for name, changes in (("_slots_by_key", update.slot_changes),
                              ("_target_ready_shards_by_slot",
                               update.target_ready_changes),
                              ("_credits_by_key", update.credit_changes)):
            apply_row_changes(getattr(self, name), changes)
        for key, value in update.target_slot_lineage_changes.items():
            self._target_slot_lineage[key] = value
        for obligation_id, value in update.obligation_changes.items():
            old = self._obligations_by_id.get(obligation_id)
            if old is not None:
                segment_ids = self._obligations_by_segment.get(
                    old["segment_id"])
                if segment_ids is not None:
                    segment_ids.discard(obligation_id)
                    if not segment_ids:
                        self._obligations_by_segment.pop(
                            old["segment_id"], None)
            if value is _ROW_TOMBSTONE:
                self._obligations_by_id.pop(obligation_id, None)
            else:
                self._obligations_by_id[obligation_id] = value
                self._obligations_by_segment.setdefault(
                    value["segment_id"], set()).add(obligation_id)

        for key, delta in update.aggregate_deltas.get(
                "source_release_watermarks", {}).items():
            if self._source_release_watermarks.get(key, 0):
                self._release_watermark_keys_by_domain.setdefault(
                    key[0], set()).add(key)
        for domain in {
                self._descriptors[segment].identity.buffer_domain_id
                for segment in update.descriptor_retirements
                if update.aggregate_deltas.get(
                    "active_descriptor_by_domain", {}).get(
                        self._descriptors[segment].identity.buffer_domain_id, 0)
                        < 0
                and self._active_descriptor_count_by_domain.get(
                    self._descriptors[segment].identity.buffer_domain_id, 0) == 0}:
            for key in self._release_watermark_keys_by_domain.pop(domain, set()):
                self._source_release_watermarks.pop(key, None)

        for segment, delta in update.active_endpoint_deltas.items():
            value = self._active_endpoint_counts.get(segment, 0) + delta
            if value:
                self._active_endpoint_counts[segment] = value
            else:
                self._active_endpoint_counts.pop(segment, None)

        self._active_resources = update.resources
        for segment in update.descriptor_retirements:
            self._cursor_runs_by_segment.pop(segment, None)
            self._cursor_versions_by_segment.pop(segment, None)
            self._cursor_unissued_by_segment.pop(segment, None)
            self._active_endpoint_counts.pop(segment, None)
            self._ack_pins_by_segment.pop(segment, None)
            self._selector_indexes.pop(segment, None)
        self.epoch = parsed.epoch
        self.initialized = True
        self.delivery_sequence = parsed.delivery_sequence
        self.watermark = parsed.seq_end
        self.state_version = parsed.result_version
        self.last_tick = parsed.tick
        self.last_replay_token = parsed.replay_token
        self._last_content = parsed.canonical_bytes
        self._snapshot_state_cache = None
        self._pricing_snapshot_cache = None
        return PipelineLedgerReceipt(parsed.snapshot_kind,
            parsed.delivery_sequence, parsed.event_count, parsed.base_watermark,
            self.watermark, parsed.base_version, self.state_version, False)

    def apply_envelope(self, envelope: Any) -> PipelineLedgerReceipt:
        """Standalone validate-and-commit helper for parser/unit tests."""
        return self.commit_prepared(self.prepare_envelope(envelope))

    def snapshot_state(self) -> Mapping[str, Any]:
        """Materialize a complete offline view on explicit request only."""
        if self._snapshot_state_cache is not None:
            return self._snapshot_state_cache
        active_descriptors = {
            segment: self._descriptors[segment]
            for segment in self._active_segments
        }
        active_descriptor_state = {
            segment: self._descriptor_state[segment]
            for segment in self._active_segments
        }
        cursors = []
        for segment in sorted(self._active_segments):
            descriptor = self._descriptors[segment]
            cursors.append(MappingProxyType({
                "segment_id": segment,
                "buffer_domain_id": descriptor.identity.buffer_domain_id,
                "generation": descriptor.identity.generation,
                "cursor_version": self._cursor_versions_by_segment.get(
                    segment, self._descriptor_state[segment]["cursor_version"]),
                "as_of_tick": self.last_tick,
                "runs": tuple(
                    self._cursor_runs_by_segment.get(segment, {})[run_index]
                    for run_index in sorted(
                        self._cursor_runs_by_segment.get(segment, {}))),
            }))
        obligations = tuple(self._obligations_by_id[key]
                            for key in sorted(self._obligations_by_id))
        slots = tuple(self._slots_by_key[key] for key in sorted(self._slots_by_key))
        credits = tuple(self._credits_by_key[key]
                        for key in sorted(self._credits_by_key))
        self._snapshot_state_cache = MappingProxyType({
            "epoch": self.epoch,
            "delivery_sequence": self.delivery_sequence,
            "event_watermark": self.watermark,
            "state_version": self.state_version,
            "tick": self.last_tick,
            "descriptors": MappingProxyType(active_descriptors),
            "descriptor_state": MappingProxyType(active_descriptor_state),
            "authoritative_cursors": tuple(cursors),
            "active_resources": self._active_resources,
            "obligations": obligations,
            "slots": slots,
            "credits": credits,
        })
        return self._snapshot_state_cache

    def pricing_snapshot(self) -> Mapping[str, Any]:
        """Cheap hot-path view of current prices and version metadata only."""
        if self._pricing_snapshot_cache is None:
            self._pricing_snapshot_cache = MappingProxyType({
                "epoch": self.epoch,
                "delivery_sequence": self.delivery_sequence,
                "event_watermark": self.watermark,
                "state_version": self.state_version,
                "tick": self.last_tick,
                "active_resources": self._active_resources,
            })
        return self._pricing_snapshot_cache

    def remaining_view(self) -> Mapping[str, Any]:
        """Read-only logic view; consumers retain descriptors and bounded cursors."""
        return self.snapshot_state()


class PipelineLedgerV4(PipelineLedger):
    """Strict live v4 consumer with target-slot delta patches.

    ``PipelineLedger`` remains the exact v3 full-row decoder used by offline
    archive comparison. OnlineSchedulerBase constructs only this class, so a
    v3 envelope cannot fall back through the live path.
    """

    PROTOCOL_VERSION = 4
