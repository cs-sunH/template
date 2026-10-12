"""Strict v3 archive and v4 live pipeline ledger state/atomicity tests."""

import copy
import json
import os
import sys
import unittest
from collections import UserDict
from collections.abc import Mapping as _TestMapping
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace
from typing import Mapping as _LegacyTypingMapping
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_WL = os.path.dirname(_HERE)
for _path in (_HERE, _WL, os.path.join(_WL, "online")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from joint.compact_pipeline import PhysicalPipelineDraft  # noqa: E402
from joint.pipeline_descriptor import PipelineDescriptor  # noqa: E402
import joint.pipeline_descriptor as pipeline_descriptor_module  # noqa: E402
from joint.pipeline_identity import (  # noqa: E402
    PipelineIdentityAllocator, PipelineIdentitySeed,
)
from joint.pipeline_ledger import (  # noqa: E402
    PipelineLedger, PipelineLedgerV4, PipelineLedgerError,
    _expected_endpoint_amount,
    _initial_cursor_run,
    _parse_pipeline_runs,
)
import joint.pipeline_ledger as pipeline_ledger_module  # noqa: E402
from joint.test_pipeline_descriptor import (  # noqa: E402
    _descriptor_with_shard_count, _wire as _descriptor_wire,
)
from joint.work_ledger import WorkLedger, WorkLedgerError  # noqa: E402
from joint.work_ledger import canonical_work_id, leg_work_id  # noqa: E402
import joint.work_ledger as work_ledger_module  # noqa: E402
from online.online_scheduler_base import (  # noqa: E402
    OnlineSchedulerBase as LiveOnlineSchedulerBase,
)


class OnlineSchedulerBase(LiveOnlineSchedulerBase):
    """Test-only exact v3 archive adapter; live base remains strict v4."""

    def _apply_work_ledger(self, delta):
        payload = delta.get("work_ledger")
        parsed = self.joint_pipeline_ledger.validate_envelope(payload)
        work_tx = self.joint_work_ledger.begin_protocol_v3_transaction(
            delivery_sequence=payload.get("delivery_sequence"),
            tick=payload.get("tick"), snapshot_kind=payload.get("snapshot_kind"))
        prepared = self.joint_pipeline_ledger.prepare_envelope(
            payload, work_transaction=work_tx, _parsed=parsed)
        work_tx.commit(notify=False)
        self.joint_pipeline_ledger.commit_prepared(prepared)
        work_tx.notify()
        self.work_ledger_absent_seen = False
        return prepared.parsed.canonical_bytes


class _TestCustomMapping(_TestMapping):
    def __init__(self, values):
        self._values = dict(values)

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


class _TestDictSubclass(dict):
    pass


class _CountingTuple(tuple):
    """Observe real descriptor-run indexing versus whole-run iteration."""

    def __new__(cls, values, reads):
        result = super().__new__(cls, values)
        result._reads = reads
        return result

    def __getitem__(self, index):
        if isinstance(index, int):
            self._reads["indexed"].append(index)
        return super().__getitem__(index)

    def __iter__(self):
        self._reads["iterated"] += len(self)
        return super().__iter__()


def _v4_json_thaw(value):
    """Copy immutable accepted rows into ordinary JSON-ready fixture values."""
    if isinstance(value, _TestMapping):
        return {key: _v4_json_thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_v4_json_thaw(item) for item in value]
    return value


class _ToggleEqStr(str):
    """Exercise the parser's accepted str-subclass equality side effects."""

    def __new__(cls, value, role, visits):
        item = super().__new__(cls, value)
        item.role = role
        item.visits = visits
        return item

    def __eq__(self, other):
        if (self.role == "after"
                and getattr(other, "role", None) == "descriptor"):
            self.visits["after_expected_comparisons"] += 1
            return self.visits["after_expected_comparisons"] == 1
        return str.__eq__(self, other)

    __hash__ = str.__hash__


class _TestVirtualMapping:
    def __init__(self, values):
        self._values = dict(values)

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def keys(self):
        return self._values.keys()

    def items(self):
        return self._values.items()

    def get(self, key, default=None):
        return self._values.get(key, default)


class _CountedCopyKeys:
    """Count actual key iteration, keeping key-set comparisons separate."""

    def __init__(self, keys, name, visits):
        self._keys = keys
        self._name = name
        self._visits = visits

    def __iter__(self):
        self._visits[self._name] += 1
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def __eq__(self, other):
        if isinstance(other, _CountedCopyKeys):
            other = other._keys
        return self._keys == other


class _CountedWireMapping(_TestMapping):
    """Mapping whose copied key iterators expose real dict-copy work."""

    def __init__(self, values, name, visits):
        self._values = dict(values)
        self._name = name
        self._visits = visits

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def keys(self):
        return _CountedCopyKeys(
            self._values.keys(), self._name, self._visits)


def _test_mapping_tree(value, factory):
    if isinstance(value, dict):
        return factory({key: _test_mapping_tree(item, factory)
                        for key, item in value.items()})
    if isinstance(value, list):
        return [_test_mapping_tree(item, factory) for item in value]
    if isinstance(value, tuple):
        return tuple(_test_mapping_tree(item, factory) for item in value)
    return value


class _TestSpoofedMappingClass:
    @property
    def __class__(self):
        return _TestMapping


def _empty_envelope(kind, delivery, *, base_watermark, base_version,
                    tick=0):
    return {
        "producer": "cpp_online",
        "schema_version": 3,
        "protocol_version": 3,
        "bridge_protocol": "bridge-v3-20261009",
        "work_ledger_schema": "workledger-v3-20261009",
        "epoch": "test-run",
        "delivery_sequence": delivery,
        "base_watermark": base_watermark,
        "logical_seq_start": base_watermark + 1,
        "logical_seq_end": base_watermark,
        "logical_event_count": 0,
        "replay_token": f"cpp_online/test-run/{delivery}",
        "descriptor_version": 1,
        "base_state_version": base_version,
        "result_state_version": base_version,
        "snapshot_kind": kind,
        "tick": tick,
        "descriptor_commits": [],
        "node_transitions": [],
        "pipeline_transition_runs": [],
        "cursor_patches": [],
        "active_resources": [],
    }


def _descriptor_commit(descriptor):
    return {
        "identity": descriptor.to_wire()["identity"],
        "descriptor": descriptor.to_wire(),
        "commit_tick": 0,
        "cursor_version": 0,
        "service_done": False,
        "settle_done": False,
        "retired": False,
    }


def _pipeline_run(segment, seq, transition, tick, *, cursor_start=0,
                  cursor_end=0, obligation_start=(), obligation_end=()):
    return {
        "first_logical_seq": seq,
        "pattern_id": f"{segment}/{transition}/{seq}",
        "repeat_count": 1,
        "tile_first": None,
        "tile_stride": None,
        "rank_first": None,
        "rank_stride": None,
        "shard_first": None,
        "shard_stride": None,
        "transition_kind": transition,
        "descriptor_segment_id": segment,
        "endpoint_selector": None,
        "store_id_first": None,
        "store_id_stride": None,
        "legacy_context": None,
        "tick_first": tick,
        "tick_stride": 0,
        "resource_leg": None,
        "resource_quantity_prefix": [
            {"ops": None, "bytes": None, "latency_ns": None},
            {"ops": None, "bytes": None, "latency_ns": None},
        ],
        "cursor_version_start": cursor_start,
        "cursor_version_end": cursor_end,
        "slot_patches": [],
        "credit_patches": [],
        "obligation_patches": list(obligation_end),
    }


def _full_with_descriptor(*, node_events=(), descriptor=None):
    descriptor = descriptor or PipelineDescriptor.from_wire(_descriptor_wire())
    events = list(node_events)
    run = _pipeline_run(descriptor.identity.segment_id, 1,
                        "descriptor_commit", 0)
    segment = descriptor.identity.segment_id
    cursor_rows = []
    obligation_rows = []
    for run_index, endpoint_run in enumerate(descriptor.endpoint_runs):
        cursor = _initial_cursor_run(descriptor, segment, run_index)
        cursor_rows.append({"run_index": run_index, "value": cursor})
        remaining = (endpoint_run.tile_count * len(endpoint_run.shard_ordinals)
                     * len(endpoint_run.legs) - cursor["completed_endpoint_count"])
        if remaining:
            obligation_rows.append({
                "obligation_id": f"{segment}/run/{run_index}",
                "segment_id": segment, "rank": endpoint_run.rank,
                "first_store_id": cursor["next_store_id"],
                "remaining_endpoint_count": remaining, "state": "pending",
            })
    run["obligation_patches"] = [
        {"before": None, "after": row} for row in sorted(
            obligation_rows, key=lambda item: item["obligation_id"])]
    count = 1 + len(events)
    envelope = _empty_envelope(
        "full", 0, base_watermark=0, base_version=0)
    envelope.update({
        "logical_seq_end": count,
        "logical_event_count": count,
        "result_state_version": 1,
        "descriptor_commits": [_descriptor_commit(descriptor)],
        "node_transitions": events,
        "pipeline_transition_runs": [run],
        "cursor_patches": [{
            "segment_id": segment, "cursor_version": 0, "retired": False,
            "runs": cursor_rows,
        }],
    })
    return descriptor, envelope


def _use_v4_identity(envelope):
    envelope = copy.deepcopy(envelope)
    envelope.update({
        "schema_version": 4,
        "protocol_version": 4,
        "bridge_protocol": "bridge-v4-20261011",
        "work_ledger_schema": "workledger-v4-20261011",
    })
    return envelope


def _v4_target_delta(key, before_generation, predecessor_seq,
                     predecessor_count, operations):
    return {
        "form": "target_delta",
        "key": {
            "buffer_domain_id": key[0], "role": "target",
            "rank": key[2], "shard_ordinal": None,
            "slot_index": key[4],
        },
        "before_slot_generation": before_generation,
        "predecessor_logical_seq": predecessor_seq,
        "predecessor_consumed_count": predecessor_count,
        "operations": copy.deepcopy(operations),
    }


def _v4_full_with_descriptor(descriptor):
    descriptor, frame = _full_with_descriptor(descriptor=descriptor)
    frame = _use_v4_identity(frame)
    commit = frame["pipeline_transition_runs"][0]
    for slot_index in (0, 1):
        key = (descriptor.identity.buffer_domain_id, "target",
               descriptor.identity.rank, None, slot_index)
        commit["slot_patches"].append(_v4_target_delta(
            key, 0, 0, 0, [{"op": "initialize", "slot_generation": 0}]))
    return descriptor, frame


def _v4_issue_delta(descriptor, *, delivery, watermark, issue_count):
    frame = _use_v4_identity(_issue_delta(
        descriptor, delivery=delivery, watermark=watermark,
        issue_count=issue_count))
    lineages = {
        (descriptor.identity.buffer_domain_id, "target",
         descriptor.identity.rank, None, slot): (0, 1, 0)
        for slot in (0, 1)
    }
    for transition in frame["pipeline_transition_runs"]:
        converted = []
        for patch in transition["slot_patches"]:
            before, after = patch["before"], patch["after"]
            row = after if after is not None else before
            if row["role"] == "source":
                converted.append(patch)
                continue
            key = (row["buffer_domain_id"], "target", row["rank"], None,
                   row["slot_index"])
            generation, predecessor, count = lineages[key]
            owner = row["owner"]
            group = next(item for item in descriptor.groups
                         if item.group_ordinal == owner["group_ordinal"])
            target_shards = []
            for shard_ordinal in range(len(group.shards)):
                _global, shard = descriptor.selected_shard_tile(
                    owner["group_ordinal"], owner["tile_ordinal"] - 1,
                    shard_ordinal)
                if (shard.emit_endpoint
                        and shard.target_rank == descriptor.identity.rank):
                    target_shards.append(shard_ordinal)
            sequence = transition["first_logical_seq"]
            converted.append(_v4_target_delta(
                key, generation, predecessor, count, [{
                    "op": "claim", "slot_generation": generation + 1,
                    "owner": owner,
                    "target_shards": target_shards,
                }]))
            lineages[key] = (generation + 1, sequence, 0)
        transition["slot_patches"] = converted
    return frame


def _v4_partitioned_consume_descriptor(shard_count, *, shared_terminal):
    descriptor = _descriptor_with_shard_count(shard_count)
    wire = descriptor.to_wire()
    base = wire["consume_completion_runs"][0]
    runs = []
    for shard_ordinal in range(shard_count):
        row = copy.deepcopy(base)
        row["shard_ordinals"] = [shard_ordinal]
        if shared_terminal:
            row["first_dependency_json_id"] = base[
                "first_dependency_json_id"]
            row["dependency_shard_stride"] = 0
        else:
            row["first_dependency_json_id"] = (
                base["first_dependency_json_id"] + 10 * shard_ordinal)
            row["dependency_shard_stride"] = 1
        runs.append(row)
    wire["consume_completion_runs"] = runs
    return PipelineDescriptor.from_wire(wire)


def _v4_all_target_endpoint_shards_descriptor(shard_count):
    """Validated descriptor with one data_recv cursor run over all shards."""
    wire = copy.deepcopy(_descriptor_wire())
    wire["groups"][0]["shards"] = [{
        "shard_ordinal": shard,
        "source_rank": 0,
        "target_rank": 1,
        "total_data_bytes": 5,
        "bytes_per_unit": 1,
        "noc_path": [0, 1],
    } for shard in range(shard_count)]
    wire["tag_allocation_runs"][0]["count"] = 3 * shard_count
    wire["tag_allocation_runs"][0]["shard_count"] = shard_count
    source_run, target_run = wire["endpoint_runs"]
    source_run["legs"][0]["tag_tile_stride"] = shard_count
    target_run["shard_ordinals"] = list(range(shard_count))
    target_run["tile_stride"] = shard_count
    target_run["shard_stride"] = 1
    target_run["legs"][0]["tag_tile_stride"] = shard_count
    target_run["legs"][0]["tag_shard_stride"] = 1
    wire["logical_id_ranges"] = [
        wire["logical_id_ranges"][0],
        {
            "rank": 1,
            "first_json_id": 55,
            "count": 3 * shard_count - 1,
            "stride": 1,
            "kind": "virtual_endpoint_hole",
        },
    ]
    wire["consume_completion_runs"][0]["shard_ordinals"] = list(
        range(shard_count))
    wire["consume_completion_runs"][0]["first_dependency_json_id"] = 1000
    wire["consume_completion_runs"][0]["dependency_shard_stride"] = 1
    return PipelineDescriptor.from_wire(wire)


def _v4_descriptor_same_domain_next_segment(descriptor, ordinal, shift):
    wire = descriptor.to_wire()
    identity = wire["identity"]
    identity["segment_ordinal"] = ordinal
    identity["segment_id"] = (
        descriptor.identity.segment_id.rsplit("/", 1)[0]
        + "/" + str(ordinal))
    for row in wire["logical_id_ranges"]:
        row["first_json_id"] += shift
    for row in wire["endpoint_runs"]:
        row["first_json_id"] += shift
        if row["initial_parent_json_id"] is not None:
            row["initial_parent_json_id"] += shift
    for field in ("endpoint_dependency_runs", "consume_completion_runs"):
        for row in wire[field]:
            row["first_dependency_json_id"] += shift
    for row in wire["tag_allocation_runs"]:
        row["first_tag"] += ordinal * 10
        row["emission_ordinal"] += ordinal
    for row in wire["endpoint_runs"]:
        for leg in row["legs"]:
            if leg["tag_emission_ordinal"] is not None:
                leg["tag_emission_ordinal"] += ordinal
    for row in wire["anchors"]:
        if row["node_id"] is not None:
            row["node_id"] += shift
        selector = row.get("selector")
        if (selector is not None
                and "tag_allocation_emission_ordinal" in selector):
            selector["tag_allocation_emission_ordinal"] += ordinal
    return PipelineDescriptor.from_wire(wire)


def _v4_endpoint_transition(descriptor, run_index, sequence, kind,
                            cursor_start, cursor_end, tick):
    run = descriptor.endpoint_runs[run_index]
    shard = run.shard_ordinals[0]
    leg_index = 0
    endpoint = run.legs[leg_index].endpoint
    selector = {
        "group_ordinal": run.group_ordinal,
        "tile_ordinal": run.first_tile,
        "shard_ordinal": shard,
        "endpoint": endpoint,
    }
    store_id = descriptor.endpoint_id(
        run_index, run.first_tile, shard, leg_index) + 1
    amount = _expected_endpoint_amount(descriptor, run_index, 0)
    transition = _pipeline_run(
        descriptor.identity.segment_id, sequence, kind, tick,
        cursor_start=cursor_start, cursor_end=cursor_end)
    transition.update({
        "tile_first": run.first_tile, "tile_stride": 0,
        "rank_first": run.rank, "rank_stride": 0,
        "shard_first": shard, "shard_stride": 0,
        "endpoint_selector": selector,
        "store_id_first": store_id, "store_id_stride": 0,
        "legacy_context": {
            "request_id": run.legacy_context.request_id,
            "stage": run.legacy_context.stage,
            "generation": run.legacy_context.generation,
        },
        "resource_leg": "endpoint",
        "resource_quantity_prefix": [
            {field: 0 if value is not None else None
             for field, value in amount.items()}, amount],
    })
    return transition, selector, store_id, amount


def _v4_single_target_issue(descriptor, *, delivery, watermark,
                            base_version, sequence, cursor_start,
                            slot_generation, predecessor_seq):
    run_index = next(
        index for index, run in enumerate(descriptor.endpoint_runs)
        if run.rank == descriptor.identity.rank
        and run.legs[0].endpoint == "data_recv")
    run = descriptor.endpoint_runs[run_index]
    transition, selector, store_id, _amount = _v4_endpoint_transition(
        descriptor, run_index, sequence, "tile_issue", cursor_start,
        cursor_start + 1, 1)
    shard = run.shard_ordinals[0]
    target_key = (descriptor.identity.buffer_domain_id, "target",
                  descriptor.identity.rank, None,
                  (descriptor.tile(selector["group_ordinal"],
                                   selector["tile_ordinal"] - 1)
                   .global_tile_ordinal - 1) % 2)
    owner = {
        "segment_id": descriptor.identity.segment_id,
        "pipeline_generation": descriptor.identity.generation,
        "stage_epoch": descriptor.identity.stage_epoch,
        "staging_allocation_id": descriptor.identity.staging_allocation_id,
        "group_ordinal": run.group_ordinal,
        "tile_ordinal": run.first_tile,
    }
    group = next(group for group in descriptor.groups
                 if group.group_ordinal == run.group_ordinal)
    target_shards = []
    for shard_ordinal in range(len(group.shards)):
        _global, shard_tile = descriptor.selected_shard_tile(
            run.group_ordinal, run.first_tile - 1, shard_ordinal)
        if (shard_tile.emit_endpoint
                and shard_tile.target_rank == descriptor.identity.rank):
            target_shards.append(shard_ordinal)
    transition["slot_patches"] = [_v4_target_delta(
        target_key, slot_generation, predecessor_seq, 0, [{
            "op": "claim", "slot_generation": slot_generation + 1,
            "owner": owner, "target_shards": target_shards,
        }])]
    active = {
        "store_id": store_id,
        "tile_ordinal": run.first_tile,
        "shard_ordinal": shard,
        "endpoint": run.legs[0].endpoint,
        "transfer_id": 9000 + sequence,
        "flow_id": None, "network_done": False, "hbm_done": False,
        "owner": owner,
    }
    cursor = _initial_cursor_run(
        descriptor, descriptor.identity.segment_id, run_index)
    next_position = 1
    endpoints_per_tile = len(run.shard_ordinals) * len(run.legs)
    next_tile_offset, within_tile = divmod(
        next_position, endpoints_per_tile)
    next_shard_index, next_leg_index = divmod(within_tile, len(run.legs))
    next_store_id = (None if next_tile_offset >= run.tile_count else
        descriptor.endpoint_id(
            run_index, run.first_tile + next_tile_offset,
            run.shard_ordinals[next_shard_index], next_leg_index) + 1)
    cursor.update({
        "remaining_endpoint_count": (
            run.tile_count * len(run.shard_ordinals) * len(run.legs)
            - cursor["completed_endpoint_count"]),
        "in_flight": True,
        "next_store_id": next_store_id,
        "tile_offset": next_tile_offset,
        "shard_index": next_shard_index,
        "leg_index": next_leg_index,
        "active_endpoints": [active],
    })
    obligation_id = f"{descriptor.identity.segment_id}/run/{run_index}"
    obligation = {
        "obligation_id": obligation_id,
        "segment_id": descriptor.identity.segment_id,
        "rank": run.rank,
        "first_store_id": _initial_cursor_run(
            descriptor, descriptor.identity.segment_id, run_index)[
                "next_store_id"],
        "remaining_endpoint_count": cursor["remaining_endpoint_count"],
        "state": "active",
    }
    transition["obligation_patches"] = [{
        "before": {**obligation, "state": "pending"},
        "after": obligation,
    }]
    frame = _use_v4_identity(_empty_envelope(
        "delta", delivery, base_watermark=watermark,
        base_version=base_version, tick=1))
    frame.update({
        "logical_seq_start": sequence,
        "logical_seq_end": sequence,
        "logical_event_count": 1,
        "result_state_version": base_version + 1,
        "pipeline_transition_runs": [transition],
        "cursor_patches": [{
            "segment_id": descriptor.identity.segment_id,
            "cursor_version": cursor_start + 1,
            "retired": False,
            "runs": [{"run_index": run_index, "value": cursor}],
        }],
    })
    return frame, target_key, run_index, active


def _v4_seed_completed_target_owner(ledger, descriptor, target_key,
                                    target_run_index):
    """Seed a coherent accepted proof state for consumer-only scale tests.

    The real claim and target cursor start come from a committed V4 issue
    envelope. This helper completes the descriptor's target cursor and seeds
    its accepted ready/terminal proof indexes, without claiming to model the
    producer's physical event tape.
    """
    segment = descriptor.identity.segment_id
    run = descriptor.endpoint_runs[target_run_index]
    target_row = ledger._slots_by_key[target_key]
    owner = target_row["owner"]
    owner_tag = pipeline_ledger_module._target_owner_tag(owner)
    total = run.tile_count * len(run.shard_ordinals) * len(run.legs)

    cursor_key = (segment, target_run_index)
    cursor = dict(ledger._cursor_runs_by_segment[segment][target_run_index])
    active = tuple(cursor["active_endpoints"])
    cursor.update({
        "completed_endpoint_count": total,
        "cancelled_endpoint_count": 0,
        "remaining_endpoint_count": 0,
        "in_flight": False,
        "next_store_id": None,
        "tile_offset": run.tile_count,
        "shard_index": 0,
        "leg_index": 0,
        "completed_out_of_order": (),
        "active_endpoints": (),
    })
    ledger._cursor_runs_by_segment[segment][target_run_index] = (
        pipeline_ledger_module._freeze_dto(cursor))

    active_count = ledger._active_endpoint_counts.get(segment, 0) - len(active)
    if active_count:
        ledger._active_endpoint_counts[segment] = active_count
    else:
        ledger._active_endpoint_counts.pop(segment, None)
    transfer_rows = ledger._active_transfers_by_run.pop(cursor_key, {})
    for endpoint in transfer_rows.values():
        transfer_id = endpoint["transfer_id"]
        count = ledger._active_transfer_id_counts.get(transfer_id, 0) - 1
        if count:
            ledger._active_transfer_id_counts[transfer_id] = count
        else:
            ledger._active_transfer_id_counts.pop(transfer_id, None)
            ledger._active_transfer_ids.discard(transfer_id)

    obligation_id = f"{segment}/run/{target_run_index}"
    obligation = ledger._obligations_by_id.pop(obligation_id, None)
    if obligation is not None:
        remaining = (ledger._obligation_remaining_by_segment.get(segment, 0)
                     - obligation["remaining_endpoint_count"])
        if remaining:
            ledger._obligation_remaining_by_segment[segment] = remaining
        else:
            ledger._obligation_remaining_by_segment.pop(segment, None)
        obligation_ids = ledger._obligations_by_segment.get(segment)
        if obligation_ids is not None:
            obligation_ids.discard(obligation_id)
            if not obligation_ids:
                ledger._obligations_by_segment.pop(segment, None)

    ledger._cursor_unissued_by_segment[segment] = sum(
        endpoint_run.tile_count * len(endpoint_run.shard_ordinals)
        * len(endpoint_run.legs)
        - pipeline_ledger_module._cursor_position_ordinal(
            ledger._cursor_runs_by_segment[segment][index], endpoint_run)
        for index, endpoint_run in enumerate(descriptor.endpoint_runs))

    for shard in target_row["target_shards_to_consume"]:
        ready_key = (target_key[0], target_key[2], target_key[4], shard)
        ledger._target_ready_shards_by_slot[ready_key] = owner_tag

    terminal_root = ledger._completed_terminal_intervals_by_rank.get(run.rank)
    completion_index = next(
        index for index, completion in enumerate(
            descriptor.consume_completion_runs)
        if completion.group_ordinal == run.group_ordinal
        and completion.rank == run.rank
        and completion.first_tile <= owner["tile_ordinal"]
        < completion.first_tile + completion.tile_count)
    completion = descriptor.consume_completion_runs[completion_index]
    for shard_position, shard in enumerate(completion.shard_ordinals):
        dependency_json_id = (
            completion.first_dependency_json_id
            + (owner["tile_ordinal"] - completion.first_tile)
            * completion.dependency_tile_stride
            + shard_position * completion.dependency_shard_stride)
        terminal_root = pipeline_ledger_module._completed_interval_add(
            terminal_root, dependency_json_id + 1)
    ledger._completed_terminal_intervals_by_rank[run.rank] = terminal_root
    ledger._snapshot_state_cache = None
    ledger._pricing_snapshot_cache = None


def _v4_request_close_frame(ledger, descriptor, *, delivery, watermark,
                            base_version, sequence, cursor_start,
                            target_key):
    segment = descriptor.identity.segment_id
    close_rows = []
    obligations = []
    for run_index, run in enumerate(descriptor.endpoint_runs):
        current = ledger._cursor_runs_by_segment[segment][run_index]
        row = _v4_json_thaw(current)
        total = run.tile_count * len(run.shard_ordinals) * len(run.legs)
        completed = row["completed_endpoint_count"] + sum(
            item["end_ordinal_exclusive"] - item["first_ordinal"]
            for item in row["completed_out_of_order"])
        active = row["active_endpoints"]
        unresolved = (total - completed - row["cancelled_endpoint_count"])
        row.update({
            "cancelled_endpoint_count": (
                row["cancelled_endpoint_count"] + unresolved - len(active)),
            "remaining_endpoint_count": len(active),
            "in_flight": bool(active),
            "next_store_id": None,
            "tile_offset": run.tile_count,
            "shard_index": 0, "leg_index": 0,
        })
        close_rows.append({"run_index": run_index, "value": row})
        obligation_id = f"{segment}/run/{run_index}"
        before = ledger._obligations_by_id.get(obligation_id)
        before = None if before is None else _v4_json_thaw(before)
        if active:
            after = {
                "obligation_id": obligation_id,
                "segment_id": segment,
                "rank": run.rank,
                "first_store_id": min(item["store_id"] for item in active),
                "remaining_endpoint_count": len(active),
                "state": "active",
            }
        else:
            after = None
        if before != after:
            obligations.append({"before": before, "after": after})

    target_row = ledger._slots_by_key[target_key]
    generation, predecessor_seq, predecessor_count = (
        ledger._target_slot_lineage[target_key])
    transition = _pipeline_run(
        segment, sequence, "request_close", 1,
        cursor_start=cursor_start, cursor_end=cursor_start + 1)
    transition["slot_patches"] = [_v4_target_delta(
        target_key, generation, predecessor_seq, predecessor_count, [{
            "op": "request_close_clear",
            "slot_generation": generation,
        }])]
    transition["obligation_patches"] = sorted(
        obligations, key=lambda patch: (
            patch["before"] or patch["after"])["obligation_id"])
    frame = _use_v4_identity(_empty_envelope(
        "delta", delivery, base_watermark=watermark,
        base_version=base_version, tick=1))
    frame.update({
        "logical_seq_start": sequence,
        "logical_seq_end": sequence,
        "logical_event_count": 1,
        "result_state_version": base_version + 1,
        "pipeline_transition_runs": [transition],
        "cursor_patches": [{
            "segment_id": segment,
            "cursor_version": cursor_start + 1,
            "retired": False,
            "runs": close_rows,
        }],
    })
    return frame


def _v4_closed_endpoint_done_frame(ledger, descriptor, run_index, *,
                                   delivery, watermark, base_version,
                                   sequence, cursor_start, active):
    segment = descriptor.identity.segment_id
    run = descriptor.endpoint_runs[run_index]
    transition, _selector, _store, _amount = _v4_endpoint_transition(
        descriptor, run_index, sequence, "endpoint_done", cursor_start,
        cursor_start + 1, 1)
    current = ledger._cursor_runs_by_segment[segment][run_index]
    final = _v4_json_thaw(current)
    final.update({
        "completed_endpoint_count": current["completed_endpoint_count"] + 1,
        "remaining_endpoint_count": 0,
        "in_flight": False,
        "next_store_id": None,
        "tile_offset": run.tile_count,
        "shard_index": 0, "leg_index": 0,
        "completed_out_of_order": (),
        "active_endpoints": [],
    })
    transition["obligation_patches"] = [{
        "before": _v4_json_thaw(
            ledger._obligations_by_id[f"{segment}/run/{run_index}"]),
        "after": None,
    }]
    frame = _use_v4_identity(_empty_envelope(
        "delta", delivery, base_watermark=watermark,
        base_version=base_version, tick=1))
    frame.update({
        "logical_seq_start": sequence,
        "logical_seq_end": sequence,
        "logical_event_count": 1,
        "result_state_version": base_version + 1,
        "pipeline_transition_runs": [transition],
        "cursor_patches": [{
            "segment_id": segment,
            "cursor_version": cursor_start + 1,
            "retired": False,
            "runs": [{"run_index": run_index, "value": final}],
        }],
    })
    return frame


def _v4_endpoint_done_frame(ledger, descriptor, run_index, *, delivery,
                            watermark, base_version, sequence,
                            cursor_start, tick=2):
    segment = descriptor.identity.segment_id
    run = descriptor.endpoint_runs[run_index]
    transition, _selector, _store, _amount = _v4_endpoint_transition(
        descriptor, run_index, sequence, "endpoint_done", cursor_start,
        cursor_start + 1, tick)
    current = ledger._cursor_runs_by_segment[segment][run_index]
    final = _v4_json_thaw(current)
    completed = current["completed_endpoint_count"] + 1
    next_store = (None if current["tile_offset"] >= run.tile_count else
        descriptor.endpoint_id(
            run_index, run.first_tile + current["tile_offset"],
            run.shard_ordinals[current["shard_index"]],
            current["leg_index"]) + 1)
    final.update({
        "completed_endpoint_count": completed,
        "remaining_endpoint_count": (run.tile_count * len(run.shard_ordinals)
            * len(run.legs) - completed
            - current["cancelled_endpoint_count"]),
        "in_flight": False,
        "next_store_id": next_store,
        "active_endpoints": [],
    })
    obligation_id = f"{segment}/run/{run_index}"
    before = _v4_json_thaw(ledger._obligations_by_id[obligation_id])
    after = dict(before)
    after.update({
        "first_store_id": next_store,
        "remaining_endpoint_count": final["remaining_endpoint_count"],
        "state": "pending",
    })
    transition["obligation_patches"] = [{"before": before, "after": after}]
    frame = _use_v4_identity(_empty_envelope(
        "delta", delivery, base_watermark=watermark,
        base_version=base_version, tick=tick))
    frame.update({
        "logical_seq_start": sequence,
        "logical_seq_end": sequence,
        "logical_event_count": 1,
        "result_state_version": base_version + 1,
        "pipeline_transition_runs": [transition],
        "cursor_patches": [{
            "segment_id": segment,
            "cursor_version": cursor_start + 1,
            "retired": False,
            "runs": [{"run_index": run_index, "value": final}],
        }],
    })
    return frame


def _v4_data_ready_frame(descriptor, run_index, *, delivery, watermark,
                         base_version, sequence, cursor_version, tick=2):
    run = descriptor.endpoint_runs[run_index]
    transition = _pipeline_run(
        descriptor.identity.segment_id, sequence, "data_ready", tick,
        cursor_start=cursor_version, cursor_end=cursor_version)
    transition.update({
        "tile_first": run.first_tile, "tile_stride": 0,
        "rank_first": run.rank, "rank_stride": 0,
        "shard_first": run.shard_ordinals[0], "shard_stride": 0,
        "endpoint_selector": {
            "group_ordinal": run.group_ordinal,
            "tile_ordinal": run.first_tile,
            "shard_ordinal": run.shard_ordinals[0],
            "endpoint": run.legs[0].endpoint,
        },
        "store_id_first": descriptor.endpoint_id(
            run_index, run.first_tile, run.shard_ordinals[0], 0) + 1,
        "store_id_stride": 0,
        "legacy_context": {
            "request_id": run.legacy_context.request_id,
            "stage": run.legacy_context.stage,
            "generation": run.legacy_context.generation,
        },
    })
    frame = _use_v4_identity(_empty_envelope(
        "delta", delivery, base_watermark=watermark,
        base_version=base_version, tick=tick))
    frame.update({
        "logical_seq_start": sequence, "logical_seq_end": sequence,
        "logical_event_count": 1, "result_state_version": base_version + 1,
        "pipeline_transition_runs": [transition],
    })
    return frame


def _v4_consume_frame(descriptor, *, delivery, watermark, base_version,
                      predecessor_seq, predecessor_count, slot_generation,
                      target_key, operations, cursor_version=2,
                      first_sequence=None, tick=2):
    sequence = watermark + 1 if first_sequence is None else first_sequence
    transition = _pipeline_run(
        descriptor.identity.segment_id, sequence, "consume_done", tick,
        cursor_start=cursor_version, cursor_end=cursor_version)
    transition["slot_patches"] = [_v4_target_delta(
        target_key, slot_generation, predecessor_seq, predecessor_count,
        operations)]
    frame = _use_v4_identity(_empty_envelope(
        "delta", delivery, base_watermark=watermark,
        base_version=base_version, tick=tick))
    frame.update({
        "logical_seq_start": sequence,
        "logical_seq_end": sequence,
        "logical_event_count": 1,
        "result_state_version": base_version + 1,
        "pipeline_transition_runs": [transition],
    })
    return frame


def _v4_consume_runs_frame(descriptor, *, delivery, watermark, base_version,
                           first_sequence, predecessor_seq,
                           predecessor_count, slot_generation, target_key,
                           cursor_version, operations, tick=2):
    transitions = []
    for offset, operation in enumerate(operations):
        sequence = first_sequence + offset
        transition = _pipeline_run(
            descriptor.identity.segment_id, sequence, "consume_done", tick,
            cursor_start=cursor_version, cursor_end=cursor_version)
        transition["slot_patches"] = [_v4_target_delta(
            target_key, slot_generation, predecessor_seq, predecessor_count,
            [operation])]
        transitions.append(transition)
        predecessor_seq = sequence
        predecessor_count += len(operation["add_shards"])
    frame = _use_v4_identity(_empty_envelope(
        "delta", delivery, base_watermark=watermark,
        base_version=base_version, tick=tick))
    frame.update({
        "logical_seq_start": first_sequence,
        "logical_seq_end": first_sequence + len(transitions) - 1,
        "logical_event_count": len(transitions),
        "result_state_version": base_version + 1,
        "pipeline_transition_runs": transitions,
    })
    return frame


def _two_group_physical_descriptor():
    """Build real source/target endpoint runs over two global tile ranges."""
    identity = PipelineIdentityAllocator().preview_batch((
        PipelineIdentitySeed("multi-group", "decode", 1, 4, 1, 9),))[0]
    draft = PhysicalPipelineDraft(
        identity=identity, stream_kind="decode_remote_credit")
    shard = SimpleNamespace(source_rank=0, target_rank=1, noc_path=(0, 1))
    for ordinal, global_first in enumerate((1, 4)):
        group = draft.add_group_range(
            pass_or_window_ordinal=ordinal, source_ordinal=ordinal,
            source_key=f"source-{ordinal}", layer_start=ordinal * 2,
            layer_end=(ordinal + 1) * 2,
            global_tile_first=global_first, tile_count=3,
            axis_unit="stream_bytes", total_units=3,
            units_per_tile=1, tail_units=1, owner_shards=(shard,),
            shard_total_bytes=(3,))
        emission = ordinal
        first_tag = 1 + ordinal * 6
        first_id = ordinal * 6
        draft.add_tag_allocation_reference(
            allocator_scope_id="graph-batch-transfer-tags-v1",
            emission_ordinal=emission, first_tag=first_tag, count=6,
            group_ordinal=group, first_tile=1, tile_count=3,
            first_shard=0, shard_count=1)
        tag_fields = {
            "tag_emission_ordinal": emission,
            "tag_offset": 0,
            "tag_tile_stride": 2,
            "tag_shard_stride": 1,
        }
        source_run = {
            "rank": 0, "group_ordinal": group,
            "first_tile": 1, "tile_count": 3,
            "shard_ordinals": [0], "first_json_id": first_id,
            "tile_stride": 2, "shard_stride": 2,
            "initial_parent_json_id": None,
            "legacy_context": {"request_id": "multi-group",
                               "stage": "decode", "generation": 1},
            "legs": [
                {"endpoint": "data_send", "json_offset": 0,
                 **tag_fields},
                {"endpoint": "fill_ack_recv", "json_offset": 1,
                 **{**tag_fields, "tag_offset": 1}},
            ],
        }
        target_run = {
            **source_run,
            "rank": 1,
            "legs": [
                {"endpoint": "data_recv", "json_offset": 0,
                 **tag_fields},
                {"endpoint": "fill_ack_send", "json_offset": 1,
                 **{**tag_fields, "tag_offset": 1}},
            ],
        }
        draft.add_endpoint_run(
            source_run,
            anchor_exceptions=((1, 0, "data_send", "start"),)
            if ordinal == 0 else ())
        draft.add_endpoint_run(
            target_run,
            anchor_exceptions=((3, 0, "fill_ack_send", "frontier"),)
            if ordinal == 1 else ())
        draft.add_consume_completion_run({
            "rank": 1, "group_ordinal": group,
            "first_tile": 1, "tile_count": 3,
            "shard_ordinals": [0],
            "dependency_kind": "virtual_endpoint_terminal",
            "first_dependency_json_id": first_id,
            "dependency_tile_stride": 2,
            "dependency_shard_stride": 0,
        })
    return draft.finish()


def _full_tile_endpoint_amount_reference(descriptor, run_index, ordinal):
    """Test oracle using the old full-tile shape and explicit shard lookup."""
    run = descriptor.endpoint_runs[run_index]
    per_tile = len(run.shard_ordinals) * len(run.legs)
    tile_offset, within_tile = divmod(ordinal, per_tile)
    shard_position, leg_index = divmod(within_tile, len(run.legs))
    tile = descriptor.tile(
        run.group_ordinal, run.first_tile + tile_offset - 1)
    shard = next(item for item in tile.shards
                 if item.shard_ordinal == run.shard_ordinals[shard_position])
    endpoint = run.legs[leg_index].endpoint
    if endpoint == "body_marker":
        return {"ops": 1, "bytes": 1, "latency_ns": None}
    if endpoint in ("fill_ack_send", "fill_ack_recv"):
        amount = descriptor.protocol.fill_ack_bytes
    elif endpoint in ("release_ack_send", "release_ack_recv"):
        amount = descriptor.protocol.release_ack_bytes
    else:
        amount = shard.data_bytes
    return {"ops": None, "bytes": amount, "latency_ns": None}


class SelectedShardLedgerConsumerTests(unittest.TestCase):
    def test_target_slot_validation_constructs_only_listed_shards_at_scales(self):
        original_builder = (
            pipeline_descriptor_module._make_pipeline_shard_tile)
        for shard_count in (8, 32, 128):
            descriptor = _descriptor_with_shard_count(shard_count)
            identity = descriptor.identity
            group = descriptor.groups[0]
            tile_ordinal = group.tile_axis.tile_count
            target_shard_ordinal = shard_count - 1
            global_tile_ordinal, selected = descriptor.selected_shard_tile(
                group.group_ordinal, tile_ordinal - 1,
                target_shard_ordinal)
            self.assertTrue(selected.emit_endpoint)
            self.assertEqual(selected.target_rank, identity.rank)
            row = {
                "buffer_domain_id": identity.buffer_domain_id,
                "role": "target",
                "rank": identity.rank,
                "shard_ordinal": None,
                "slot_index": (global_tile_ordinal - 1) % 2,
                "occupied": True,
                "owner": {
                    "segment_id": identity.segment_id,
                    "pipeline_generation": identity.generation,
                    "stage_epoch": identity.stage_epoch,
                    "staging_allocation_id": identity.staging_allocation_id,
                    "group_ordinal": group.group_ordinal,
                    "tile_ordinal": tile_ordinal,
                },
                "consumed": False,
                "target_shards_to_consume": [target_shard_ordinal],
                "consumed_target_shards": [],
            }

            old_built = []

            def counted_old_builder(shard, data_bytes):
                old_built.append(shard.shard_ordinal)
                return original_builder(shard, data_bytes)

            with mock.patch.object(
                    pipeline_descriptor_module,
                    "_make_pipeline_shard_tile",
                    new=counted_old_builder):
                full_tile = descriptor.tile(
                    group.group_ordinal, tile_ordinal - 1)
                old_expected_targets = tuple(sorted(
                    shard.shard_ordinal for shard in full_tile.shards
                    if shard.emit_endpoint
                    and shard.target_rank == identity.rank))
            self.assertEqual(len(old_built), shard_count)
            self.assertIn(target_shard_ordinal, old_expected_targets)

            selected_built = []

            def counted_selected_builder(shard, data_bytes):
                selected_built.append(shard.shard_ordinal)
                return original_builder(shard, data_bytes)

            with mock.patch.object(
                    pipeline_descriptor_module,
                    "_make_pipeline_shard_tile",
                    new=counted_selected_builder):
                parsed = pipeline_ledger_module._parse_slot_rows(
                    [row], "slot_patches", {identity.segment_id: descriptor})
            self.assertEqual(selected_built, [target_shard_ordinal])
            self.assertEqual(
                parsed[0]["target_shards_to_consume"],
                (target_shard_ordinal,))

            invalid = dict(row)
            invalid["target_shards_to_consume"] = [shard_count]
            invalid_built = []

            def counted_invalid_builder(shard, data_bytes):
                invalid_built.append(shard.shard_ordinal)
                return original_builder(shard, data_bytes)

            with mock.patch.object(
                    pipeline_descriptor_module,
                    "_make_pipeline_shard_tile",
                    new=counted_invalid_builder):
                with self.assertRaisesRegex(
                        PipelineLedgerError,
                        "target consumption shards disagree with descriptor tile"):
                    pipeline_ledger_module._parse_slot_rows(
                        [invalid], "slot_patches",
                        {identity.segment_id: descriptor})
            self.assertEqual(invalid_built, [])
            print(
                "[target-slot-selected-geometry-scale] "
                f"S={shard_count} old_full_tile_builds={len(old_built)} "
                f"selected_target_builds={len(selected_built)}")

        descriptor = _two_group_physical_descriptor()
        identity = descriptor.identity
        group = descriptor.groups[1]
        tile_ordinal = group.tile_axis.tile_count
        global_tile_ordinal, selected = descriptor.selected_shard_tile(
            group.group_ordinal, tile_ordinal - 1, 0)
        self.assertTrue(selected.emit_endpoint)
        row = {
            "buffer_domain_id": identity.buffer_domain_id,
            "role": "target",
            "rank": identity.rank,
            "shard_ordinal": None,
            "slot_index": (global_tile_ordinal - 1) % 2,
            "occupied": True,
            "owner": {
                "segment_id": identity.segment_id,
                "pipeline_generation": identity.generation,
                "stage_epoch": identity.stage_epoch,
                "staging_allocation_id": identity.staging_allocation_id,
                "group_ordinal": group.group_ordinal,
                "tile_ordinal": tile_ordinal,
            },
            "consumed": False,
            "target_shards_to_consume": [0],
            "consumed_target_shards": [],
        }
        group_built = []

        def counted_group_builder(shard, data_bytes):
            group_built.append(shard.shard_ordinal)
            return original_builder(shard, data_bytes)

        with mock.patch.object(
                pipeline_descriptor_module, "_make_pipeline_shard_tile",
                new=counted_group_builder):
            parsed_group = pipeline_ledger_module._parse_slot_rows(
                [row], "slot_patches",
                {identity.segment_id: descriptor})
        self.assertEqual(group_built, [0])
        self.assertEqual(parsed_group[0]["slot_index"],
                         (global_tile_ordinal - 1) % 2)
        released_global_tile = (
            global_tile_ordinal - descriptor.protocol.release_lag_tiles)
        if released_global_tile > 0:
            self.assertEqual(
                (released_global_tile - 1) % descriptor.protocol.buffer_slots,
                parsed_group[0]["slot_index"])

    @staticmethod
    def _old_target_membership_contains(values, target):
        comparisons = 0
        for value in values:
            comparisons += 1
            if value == target:
                return True, comparisons
        return False, comparisons

    def test_target_membership_binary_search_and_slot_parse_cost_at_scales(self):
        class CountingSequence:
            def __init__(self, values):
                self.values = values
                self.index_reads = 0

            def __len__(self):
                return len(self.values)

            def __getitem__(self, index):
                self.index_reads += 1
                return self.values[index]

        for values, target, expected, expected_reads in (
                ((), 0, False, 0),
                ((7,), 7, True, 1),
                ((7,), 6, False, 1)):
            counted = CountingSequence(values)
            self.assertEqual(
                pipeline_ledger_module._sorted_uint_tuple_contains(
                    counted, target), expected)
            self.assertEqual(counted.index_reads, expected_reads)

        original_uint = pipeline_ledger_module._uint
        original_selected = PipelineDescriptor.selected_shard_tile
        for shard_count in (8, 32, 128):
            descriptor = _descriptor_with_shard_count(shard_count)
            identity = descriptor.identity
            group = descriptor.groups[0]
            tile_ordinal = group.tile_axis.tile_count
            global_tile_ordinal, _selected = descriptor.selected_shard_tile(
                group.group_ordinal, tile_ordinal - 1, 0)
            target_members = list(range(shard_count))
            row = {
                "buffer_domain_id": identity.buffer_domain_id,
                "role": "target",
                "rank": identity.rank,
                "shard_ordinal": None,
                "slot_index": (global_tile_ordinal - 1) % 2,
                "occupied": True,
                "owner": {
                    "segment_id": identity.segment_id,
                    "pipeline_generation": identity.generation,
                    "stage_epoch": identity.stage_epoch,
                    "staging_allocation_id": identity.staging_allocation_id,
                    "group_ordinal": group.group_ordinal,
                    "tile_ordinal": tile_ordinal,
                },
                "consumed": False,
                "target_shards_to_consume": target_members,
                "consumed_target_shards": [],
            }
            selector_indexes = {
                identity.segment_id:
                    pipeline_ledger_module._build_descriptor_selector_index(
                        descriptor)
            }
            parsed = pipeline_ledger_module._parse_slot_rows(
                [row], "slot_patches", {identity.segment_id: descriptor},
                selector_indexes)
            targets = parsed[0]["target_shards_to_consume"]
            self.assertEqual(targets, tuple(target_members))
            self.assertEqual(len(targets), shard_count)

            list_comparisons = 0
            binary_index_reads = 0
            old_members = list(targets)
            for target in targets:
                counted = CountingSequence(targets)
                actual = pipeline_ledger_module._sorted_uint_tuple_contains(
                    counted, target)
                expected, old_comparisons = (
                    self._old_target_membership_contains(old_members, target))
                self.assertEqual(actual, expected)
                self.assertTrue(actual)
                list_comparisons += old_comparisons
                binary_index_reads += counted.index_reads
            expected_list_comparisons = shard_count * (shard_count + 1) // 2
            self.assertEqual(list_comparisons, expected_list_comparisons)
            self.assertLess(binary_index_reads, list_comparisons)

            for absent in (shard_count, pipeline_ledger_module.U64_MAX):
                counted = CountingSequence(targets)
                actual = pipeline_ledger_module._sorted_uint_tuple_contains(
                    counted, absent)
                expected, _old_comparisons = (
                    self._old_target_membership_contains(old_members, absent))
                self.assertEqual(actual, expected)
                self.assertFalse(actual)
                self.assertLessEqual(
                    counted.index_reads, shard_count.bit_length() + 1)
            spaced = tuple(2 * index + 2
                           for index in range(shard_count))
            for absent in (spaced[0] - 1, spaced[-1] + 1):
                counted = CountingSequence(spaced)
                actual = pipeline_ledger_module._sorted_uint_tuple_contains(
                    counted, absent)
                expected, _old_comparisons = (
                    self._old_target_membership_contains(spaced, absent))
                self.assertEqual(actual, expected)
                self.assertFalse(actual)
                self.assertLessEqual(
                    counted.index_reads, shard_count.bit_length() + 1)
            print(
                "[target-membership-binary-scale] "
                f"S={shard_count} T={shard_count} "
                f"old_list_comparisons={list_comparisons} "
                f"binary_index_reads={binary_index_reads} "
                f"parsed_membership={len(targets)}")

            row_uint_reads = {
                "target_shards_to_consume": 0,
                "consumed_target_shards": 0,
            }
            selected_geometry_calls = 0

            def counted_uint(value, path, maximum=pipeline_ledger_module.U64_MAX):
                for field in row_uint_reads:
                    if path.endswith(field + "[]"):
                        row_uint_reads[field] += 1
                        break
                return original_uint(value, path, maximum)

            def counted_selected(self, group_ordinal, local_tile, shard):
                nonlocal selected_geometry_calls
                selected_geometry_calls += 1
                return original_selected(
                    self, group_ordinal, local_tile, shard)

            def parse_slot_rows(values, path):
                return pipeline_ledger_module._parse_slot_rows(
                    values, path, {identity.segment_id: descriptor},
                    selector_indexes)

            row_pair_count = 0
            current = copy.deepcopy(row)
            with mock.patch.object(
                    pipeline_ledger_module, "_uint", new=counted_uint), \
                    mock.patch.object(
                        PipelineDescriptor, "selected_shard_tile",
                        new=counted_selected):
                for consumed_count in range(shard_count):
                    before = current
                    after = copy.deepcopy(before)
                    clears_slot = consumed_count + 1 == shard_count
                    if clears_slot:
                        after.update({
                            "occupied": False,
                            "owner": None,
                            "consumed": False,
                            "target_shards_to_consume": [],
                            "consumed_target_shards": [],
                        })
                    else:
                        after["consumed_target_shards"] = list(
                            range(consumed_count + 1))
                        after["consumed"] = True
                    patch = {"before": before, "after": after}
                    parsed_patch = pipeline_ledger_module._parse_row_patches(
                        [patch], "slot_patches", parse_slot_rows,
                        pipeline_ledger_module._slot_key,
                        lambda key: (
                            key[0], key[1], key[2],
                            -1 if key[3] is None else key[3], key[4]))
                    parsed_before, parsed_after = parsed_patch[0][1:]
                    self.assertEqual(
                        parsed_before["target_shards_to_consume"], targets)
                    self.assertEqual(
                        parsed_before["consumed_target_shards"],
                        tuple(range(consumed_count)))
                    if clears_slot:
                        self.assertFalse(parsed_after["occupied"])
                        self.assertIsNone(parsed_after["owner"])
                        self.assertEqual(
                            parsed_after["target_shards_to_consume"], ())
                        self.assertEqual(
                            parsed_after["consumed_target_shards"], ())
                    else:
                        self.assertEqual(
                            parsed_after["target_shards_to_consume"], targets)
                        self.assertEqual(
                            parsed_after["consumed_target_shards"],
                            tuple(range(consumed_count + 1)))
                    row_pair_count += 1
                    current = after

            self.assertEqual(row_pair_count, shard_count)
            self.assertEqual(
                row_uint_reads["target_shards_to_consume"],
                2 * shard_count * shard_count - shard_count)
            self.assertEqual(
                row_uint_reads["consumed_target_shards"],
                shard_count * (shard_count - 1))
            self.assertEqual(
                selected_geometry_calls,
                2 * shard_count * shard_count - shard_count)
            print(
                "[target-slot-cumulative-row-parse-scale] "
                f"S={shard_count} T={row_pair_count} "
                f"target_member_uint_reads="
                f"{row_uint_reads['target_shards_to_consume']} "
                f"consumed_member_uint_reads="
                f"{row_uint_reads['consumed_target_shards']} "
                f"selected_geometry_calls={selected_geometry_calls}")

    def test_cursor_parse_and_issue_build_selected_shards_at_scales(self):
        original_builder = (
            pipeline_descriptor_module._make_pipeline_shard_tile)
        for shard_count in (8, 32, 128):
            descriptor = _descriptor_with_shard_count(shard_count)
            segment = descriptor.identity.segment_id
            _descriptor, full = _full_with_descriptor(
                descriptor=descriptor)
            old_full_tile_builds = []

            def counted_full_builder(shard, data_bytes):
                old_full_tile_builds.append(shard.shard_ordinal)
                return original_builder(shard, data_bytes)

            cursor_values = {
                item["run_index"]: item["value"]
                for item in full["cursor_patches"][0]["runs"]
            }
            with mock.patch.object(
                    pipeline_descriptor_module,
                    "_make_pipeline_shard_tile",
                    new=counted_full_builder):
                for run_index, run in enumerate(descriptor.endpoint_runs):
                    # Independent old full-tile reference for initial cursor
                    # activity and the parser's pending next endpoint check.
                    first_tile = descriptor.tile(
                        run.group_ordinal, run.first_tile - 1)
                    for shard_ordinal in run.shard_ordinals:
                        next(item for item in first_tile.shards
                             if item.shard_ordinal == shard_ordinal)
                    cursor = cursor_values[run_index]
                    if cursor["tile_offset"] < run.tile_count:
                        leg = run.legs[cursor["leg_index"]]
                        if leg.endpoint != "body_marker":
                            tile_ordinal = run.first_tile + cursor["tile_offset"]
                            next(item for item in descriptor.tile(
                                run.group_ordinal, tile_ordinal - 1).shards
                                 if item.shard_ordinal ==
                                 run.shard_ordinals[cursor["shard_index"]])

            expected_full_tile_builds = sum(
                len(descriptor.groups[run.group_ordinal].shards)
                * (1 + int(
                    cursor_values[run_index]["tile_offset"] < run.tile_count
                    and run.legs[cursor_values[run_index]["leg_index"]].endpoint
                    != "body_marker"))
                for run_index, run in enumerate(descriptor.endpoint_runs))
            self.assertEqual(
                len(old_full_tile_builds), expected_full_tile_builds)

            selected_full_builds = []

            def counted_selected_builder(shard, data_bytes):
                selected_full_builds.append(shard.shard_ordinal)
                return original_builder(shard, data_bytes)

            ledger = PipelineLedger()
            with mock.patch.object(
                    pipeline_descriptor_module,
                    "_make_pipeline_shard_tile",
                    new=counted_selected_builder):
                ledger.apply_envelope(full)
            expected_selected_full_builds = sum(
                len(run.shard_ordinals)
                + int(cursor_values[run_index]["tile_offset"] < run.tile_count
                      and run.legs[cursor_values[run_index]["leg_index"]].endpoint
                      != "body_marker")
                for run_index, run in enumerate(descriptor.endpoint_runs))
            self.assertEqual(
                len(selected_full_builds), expected_selected_full_builds)
            self.assertLess(
                len(selected_full_builds), len(old_full_tile_builds))

            issue = _issue_delta(
                descriptor, delivery=1, watermark=1, issue_count=1)
            source_run = descriptor.endpoint_runs[0]
            transition = issue["pipeline_transition_runs"][0]
            cursor_after_issue = issue["cursor_patches"][0]["runs"][0]["value"]
            next_tile = source_run.first_tile + cursor_after_issue["tile_offset"]
            old_issue_tile_builds = []

            def counted_issue_reference_builder(shard, data_bytes):
                old_issue_tile_builds.append(shard.shard_ordinal)
                return original_builder(shard, data_bytes)

            with mock.patch.object(
                    pipeline_descriptor_module,
                    "_make_pipeline_shard_tile",
                    new=counted_issue_reference_builder):
                for tile_ordinal in (
                        transition["endpoint_selector"]["tile_ordinal"],
                        next_tile, next_tile):
                    tile = descriptor.tile(
                        source_run.group_ordinal, tile_ordinal - 1)
                    next(item for item in tile.shards
                         if item.shard_ordinal
                         == source_run.shard_ordinals[0])
            self.assertEqual(
                len(old_issue_tile_builds), 3 * shard_count)

            issue_builds = []

            def counted_issue_builder(shard, data_bytes):
                issue_builds.append(shard.shard_ordinal)
                return original_builder(shard, data_bytes)

            with mock.patch.object(
                    pipeline_descriptor_module,
                    "_make_pipeline_shard_tile",
                    new=counted_issue_builder):
                ledger.apply_envelope(issue)
            self.assertEqual(len(issue_builds), 4)
            self.assertEqual(
                ledger._cursor_runs_by_segment[segment][0][
                    "active_endpoints"][0]["store_id"],
                transition["store_id_first"])
            self.assertEqual(ledger._ack_pins_by_segment.get(segment, 0), 0)
            print(
                "[cursor-selected-geometry-scale] "
                f"S={shard_count} old_cursor_full_tile_builds="
                f"{len(old_full_tile_builds)} "
                f"selected_cursor_builds={len(selected_full_builds)} "
                f"old_issue_full_tile_builds={len(old_issue_tile_builds)} "
                f"selected_issue_builds={len(issue_builds)}")

    def test_expected_endpoint_amount_matches_full_tile_oracle(self):
        # The real multigroup draft exercises nonzero global tile offsets and
        # short tails. The validated three-shard descriptor checks that the
        # selected amount remains equal to the full-tile reference when sibling
        # shard tiles exist.
        descriptors = (
            _two_group_physical_descriptor(),
            _descriptor_with_shard_count(3),
        )
        original_selected = PipelineDescriptor.selected_shard_tile
        for descriptor in descriptors:
            for run_index, run in enumerate(descriptor.endpoint_runs):
                endpoint_count = (
                    run.tile_count * len(run.shard_ordinals) * len(run.legs))
                for ordinal in range(endpoint_count):
                    with self.subTest(
                            segment=descriptor.identity.segment_id,
                            run_index=run_index, ordinal=ordinal):
                        expected = _full_tile_endpoint_amount_reference(
                            descriptor, run_index, ordinal)
                        per_tile = len(run.shard_ordinals) * len(run.legs)
                        tile_offset, within_tile = divmod(ordinal, per_tile)
                        shard_position, _leg_index = divmod(
                            within_tile, len(run.legs))
                        local_tile = run.first_tile + tile_offset - 1
                        group_shard = run.shard_ordinals[shard_position]
                        full = descriptor.tile(run.group_ordinal, local_tile)
                        selected_calls = []

                        def counted_selected(
                                self, group_ordinal, local_tile_ordinal,
                                group_shard_index):
                            result = original_selected(
                                self, group_ordinal, local_tile_ordinal,
                                group_shard_index)
                            selected_calls.append((
                                group_ordinal, local_tile_ordinal,
                                group_shard_index, result))
                            return result

                        with mock.patch.object(
                                PipelineDescriptor, "selected_shard_tile",
                                new=counted_selected):
                            actual = _expected_endpoint_amount(
                                descriptor, run_index, ordinal)

                        self.assertEqual(actual, expected)
                        self.assertEqual(len(selected_calls), 1)
                        self.assertEqual(
                            selected_calls[0][:3],
                            (run.group_ordinal, local_tile, group_shard))
                        global_tile_ordinal, shard_tile = selected_calls[0][3]
                        expected_shard = next(
                            item for item in full.shards
                            if item.shard_ordinal == group_shard)
                        self.assertEqual(global_tile_ordinal,
                                         full.global_tile_ordinal)
                        self.assertEqual(shard_tile, expected_shard)
                        slot_index = (global_tile_ordinal - 1) % 2
                        self.assertEqual(
                            slot_index,
                            (full.global_tile_ordinal - 1) % 2)
                        released_tile = (
                            global_tile_ordinal
                            - descriptor.protocol.release_lag_tiles)
                        if released_tile > 0:
                            self.assertEqual(
                                (released_tile - 1) % 2, slot_index)

    def test_endpoint_amount_ack_and_body_marker_special_cases(self):
        descriptor = _two_group_physical_descriptor()
        run = descriptor.endpoint_runs[0]
        cases = (
            ("fill_ack_send", {"ops": None, "bytes": 1,
                               "latency_ns": None}),
            ("fill_ack_recv", {"ops": None, "bytes": 1,
                               "latency_ns": None}),
            ("release_ack_send", {"ops": None, "bytes": 1,
                                  "latency_ns": None}),
            ("release_ack_recv", {"ops": None, "bytes": 1,
                                  "latency_ns": None}),
            ("body_marker", {"ops": 1, "bytes": 1,
                             "latency_ns": None}),
        )
        for endpoint, expected in cases:
            with self.subTest(endpoint=endpoint):
                changed_leg = replace(run.legs[0], endpoint=endpoint)
                changed_run = replace(run, legs=(changed_leg,))
                changed_descriptor = replace(
                    descriptor, endpoint_runs=(changed_run,))
                self.assertEqual(
                    _expected_endpoint_amount(changed_descriptor, 0, 0),
                    expected)


def _descriptor_with_ordinal(ordinal):
    """Build an independent same-request descriptor for history scaling."""
    wire = copy.deepcopy(_descriptor_wire())
    identity = wire["identity"]
    allocation = 100 + ordinal
    identity.update({
        "segment_ordinal": ordinal,
        "staging_allocation_id": allocation,
        "segment_id": f"pp1/726571/decode/1/2/1/{ordinal}",
        "buffer_domain_id": f"pb1/726571/1/{allocation}",
    })
    shift = ordinal * 100
    for row in wire["logical_id_ranges"]:
        row["first_json_id"] += shift
    for row in wire["endpoint_runs"]:
        row["first_json_id"] += shift
        if row["initial_parent_json_id"] is not None:
            row["initial_parent_json_id"] += shift
    for field in ("endpoint_dependency_runs", "consume_completion_runs"):
        for row in wire[field]:
            row["first_dependency_json_id"] += shift
    for row in wire["tag_allocation_runs"]:
        row["first_tag"] += ordinal * 10
        row["emission_ordinal"] += ordinal
    for row in wire["endpoint_runs"]:
        for leg in row["legs"]:
            if leg["tag_emission_ordinal"] is not None:
                leg["tag_emission_ordinal"] += ordinal
    for row in wire["anchors"]:
        if row["node_id"] is not None:
            row["node_id"] += shift
        selector = row.get("selector")
        if selector is not None and "tag_allocation_emission_ordinal" in selector:
            selector["tag_allocation_emission_ordinal"] += ordinal
    return PipelineDescriptor.from_wire(wire)


def _full_with_descriptors(count):
    descriptors = [_descriptor_with_ordinal(index) for index in range(count)]
    descriptors.sort(key=lambda item: item.identity.segment_id)
    commits = []
    transitions = []
    cursor_patches = []
    for seq, descriptor in enumerate(descriptors, 1):
        segment = descriptor.identity.segment_id
        commits.append(_descriptor_commit(descriptor))
        transition = _pipeline_run(segment, seq, "descriptor_commit", 0)
        cursor_rows = []
        obligations = []
        for run_index, endpoint_run in enumerate(descriptor.endpoint_runs):
            cursor = _initial_cursor_run(descriptor, segment, run_index)
            cursor_rows.append({"run_index": run_index, "value": cursor})
            remaining = (endpoint_run.tile_count * len(endpoint_run.shard_ordinals)
                         * len(endpoint_run.legs)
                         - cursor["completed_endpoint_count"])
            if remaining:
                obligations.append({
                    "obligation_id": f"{segment}/run/{run_index}",
                    "segment_id": segment, "rank": endpoint_run.rank,
                    "first_store_id": cursor["next_store_id"],
                    "remaining_endpoint_count": remaining, "state": "pending",
                })
        transition["obligation_patches"] = [
            {"before": None, "after": row} for row in sorted(
                obligations, key=lambda item: item["obligation_id"])]
        transitions.append(transition)
        cursor_patches.append({
            "segment_id": segment, "cursor_version": 0, "retired": False,
            "runs": cursor_rows,
        })
    envelope = _empty_envelope(
        "full", 0, base_watermark=0, base_version=0)
    envelope.update({
        "logical_seq_end": count,
        "logical_event_count": count,
        "result_state_version": 1,
        "descriptor_commits": commits,
        "pipeline_transition_runs": transitions,
        "cursor_patches": cursor_patches,
    })
    return descriptors[0], envelope


def _issue_delta(descriptor, *, delivery, watermark, issue_count):
    """Create real endpoint-issue work with slot, credit and obligation rows."""
    segment = descriptor.identity.segment_id
    version = 0
    current_slots = {}
    credit_before = None
    current_obligations = {}
    cursor_progress = {}
    runs = []
    for index in range(issue_count):
        run_index = index % 2
        tile_offset = index // 2
        endpoint_run = descriptor.endpoint_runs[run_index]
        tile = endpoint_run.first_tile + tile_offset
        shard = endpoint_run.shard_ordinals[0]
        endpoint = endpoint_run.legs[0].endpoint
        store_id = descriptor.endpoint_id(run_index, tile, shard, 0) + 1
        selector = {
            "group_ordinal": endpoint_run.group_ordinal,
            "tile_ordinal": tile,
            "shard_ordinal": shard,
            "endpoint": endpoint,
        }
        owner = {
            "segment_id": segment,
            "pipeline_generation": descriptor.identity.generation,
            "stage_epoch": descriptor.identity.stage_epoch,
            "staging_allocation_id": descriptor.identity.staging_allocation_id,
            "group_ordinal": endpoint_run.group_ordinal,
            "tile_ordinal": tile,
        }
        role = "source" if endpoint == "data_send" else "target"
        slot_key = (descriptor.identity.buffer_domain_id, role,
                    endpoint_run.rank,
                    shard if role == "source" else None,
                    (tile - 1) % 2)
        after_slot = {
            "buffer_domain_id": descriptor.identity.buffer_domain_id,
            "role": role,
            "rank": endpoint_run.rank,
            "shard_ordinal": shard if role == "source" else None,
            "slot_index": (tile - 1) % 2,
            "occupied": True,
            "owner": owner,
            "consumed": False,
            "target_shards_to_consume": [] if role == "source" else [shard],
            "consumed_target_shards": [],
        }
        before_slot = current_slots.get(slot_key)
        current_slots[slot_key] = after_slot
        slot_patches = [{"before": before_slot, "after": after_slot}]

        credit_patches = []
        if role == "source":
            key = (descriptor.identity.buffer_domain_id, endpoint_run.rank, shard)
            occupied_source_rows = sum(
                1 for row in current_slots.values()
                if row["role"] == "source" and row["occupied"]
                and row["rank"] == endpoint_run.rank
                and row["shard_ordinal"] == shard)
            credit_after = {
                "buffer_domain_id": descriptor.identity.buffer_domain_id,
                "source_rank": endpoint_run.rank,
                "shard_ordinal": shard,
                "capacity": 2,
                "available": 2 - occupied_source_rows,
                "outstanding_ack_count": 0,
            }
            credit_patches = [{"before": credit_before, "after": credit_after}]
            credit_before = credit_after

        active = {
            "store_id": store_id,
            "tile_ordinal": tile,
            "shard_ordinal": shard,
            "endpoint": endpoint,
            "transfer_id": 1000 + index,
            "flow_id": None,
            "network_done": False,
            "hbm_done": False,
            "owner": owner,
        }
        progress = cursor_progress.setdefault(run_index, {
            "active": [],
            "position": 0,
            "initial": _initial_cursor_run(descriptor, segment, run_index),
        })
        progress["active"].append(active)
        progress["position"] += 1
        initial = progress["initial"]
        tile_offset_after = progress["position"]
        next_store = (None if tile_offset_after >= endpoint_run.tile_count else
                      descriptor.endpoint_id(
                          run_index, endpoint_run.first_tile + tile_offset_after,
                          shard, 0) + 1)
        cursor_row = {
            **initial,
            "completed_endpoint_count": initial["completed_endpoint_count"],
            "cancelled_endpoint_count": 0,
            "remaining_endpoint_count": (
                endpoint_run.tile_count * len(endpoint_run.shard_ordinals)
                * len(endpoint_run.legs)
                - initial["completed_endpoint_count"]),
            "in_flight": True,
            "next_store_id": next_store,
            "tile_offset": tile_offset_after,
            "shard_index": 0,
            "leg_index": 0,
            "completed_out_of_order": [],
            "active_endpoints": sorted(
                progress["active"], key=lambda item: item["store_id"]),
        }
        cursor_progress[run_index]["row"] = cursor_row

        obligation_id = f"{segment}/run/{run_index}"
        first_store = descriptor.endpoint_id(
            run_index, endpoint_run.first_tile,
            endpoint_run.shard_ordinals[0], 0) + 1
        remaining = (endpoint_run.tile_count * len(endpoint_run.shard_ordinals)
                     * len(endpoint_run.legs)
                     - initial["completed_endpoint_count"])
        before_obligation = current_obligations.get(run_index)
        if before_obligation is None and run_index not in current_obligations:
            before_obligation = {
                "obligation_id": obligation_id,
                "segment_id": segment,
                "rank": endpoint_run.rank,
                "first_store_id": first_store,
                "remaining_endpoint_count": remaining,
                "state": "pending",
            }
        after_obligation = {
            "obligation_id": obligation_id,
            "segment_id": segment,
            "rank": endpoint_run.rank,
            "first_store_id": first_store,
            "remaining_endpoint_count": remaining,
            "state": "active",
        }
        obligation_patches = []
        if before_obligation["state"] != after_obligation["state"]:
            obligation_patches = [{"before": before_obligation,
                                   "after": after_obligation}]
        current_obligations[run_index] = after_obligation

        amount = _expected_endpoint_amount(descriptor, run_index,
                                           tile_offset * len(endpoint_run.legs))
        # In this descriptor each physical endpoint carries bytes only.
        transition = _pipeline_run(
            segment, watermark + index + 1, "tile_issue", 1,
            cursor_start=version, cursor_end=version + 1)
        transition.update({
            "tile_first": tile,
            "tile_stride": 0,
            "rank_first": endpoint_run.rank,
            "rank_stride": 0,
            "shard_first": shard,
            "shard_stride": 0,
            "endpoint_selector": selector,
            "store_id_first": store_id,
            "store_id_stride": 0,
            "legacy_context": {
                "request_id": endpoint_run.legacy_context.request_id,
                "stage": endpoint_run.legacy_context.stage,
                "generation": endpoint_run.legacy_context.generation,
            },
            "resource_leg": "endpoint",
            "resource_quantity_prefix": [
                {"ops": None, "bytes": 0, "latency_ns": None}, amount],
            "slot_patches": slot_patches,
            "credit_patches": credit_patches,
            "obligation_patches": obligation_patches,
        })
        runs.append(transition)
        version += 1

    cursor_patches = [{
        "segment_id": segment,
        "cursor_version": version,
        "retired": False,
        "runs": [{"run_index": run_index,
                  "value": progress["row"]}
                 for run_index, progress in sorted(cursor_progress.items())],
    }]
    envelope = _empty_envelope(
        "delta", delivery, base_watermark=watermark,
        base_version=1, tick=1)
    envelope.update({
        "logical_seq_start": watermark + 1,
        "logical_seq_end": watermark + issue_count,
        "logical_event_count": issue_count,
        "result_state_version": 2,
        "pipeline_transition_runs": runs,
        "cursor_patches": cursor_patches,
    })
    return envelope


def _header_normalized_payload_bytes(envelope):
    """Byte count for scaling analysis after only sequence metadata is fixed.

    This copy is never passed to a ledger or used for replay/canonicalization.
    Descriptor, patch, cursor, and resource payloads remain byte-for-byte as
    supplied; only sequence/watermark fields and numeric sequence suffixes in
    pattern/replay tokens are normalized to remove decimal-width noise.
    """
    normalized = copy.deepcopy(envelope)
    for field in (
            "delivery_sequence", "base_watermark", "logical_seq_start",
            "logical_seq_end", "logical_event_count"):
        if field in normalized:
            normalized[field] = 0
    if "replay_token" in normalized:
        normalized["replay_token"] = normalized["replay_token"].rsplit(
            "/", 1)[0] + "/0"
    for event in normalized.get("node_transitions", ()):
        if "logical_seq" in event:
            event["logical_seq"] = 0
    for transition in normalized.get("pipeline_transition_runs", ()):
        if "first_logical_seq" in transition:
            transition["first_logical_seq"] = 0
        pattern = transition.get("pattern_id")
        if isinstance(pattern, str) and "/" in pattern:
            transition["pattern_id"] = pattern.rsplit("/", 1)[0] + "/0"
    return len(json.dumps(
        normalized, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8"))


def _assert_obligation_prefix_total(test_case, ledger, segment):
    small_row_sum = sum(
        row["remaining_endpoint_count"]
        for row in ledger._obligations_by_id.values()
        if row["segment_id"] == segment)
    test_case.assertEqual(
        ledger._obligation_remaining_by_segment.get(segment, 0),
        small_row_sum)


def _measure_sparse_prepare(envelope, descriptor_count):
    _descriptor, full = _full_with_descriptors(descriptor_count)
    ledger = PipelineLedger()
    ledger.apply_envelope(full)

    class NoScanMapping:
        """Accepted indexes may be point-read, never enumerated per delta."""
        def __init__(self, base, name):
            self._base = base
            self._name = name

        def get(self, key, default=None):
            return self._base.get(key, default)

        def __getitem__(self, key):
            return self._base[key]

        def __contains__(self, key):
            return key in self._base

        def __len__(self):
            # O(1) metadata reads are allowed; only row enumeration is banned.
            return len(self._base)

        def _reject_scan(self, *args, **kwargs):
            raise AssertionError(
                f"sparse prepare enumerated accepted {self._name}")

        __iter__ = _reject_scan
        keys = _reject_scan
        items = _reject_scan
        values = _reject_scan
        copy = _reject_scan

    class NoScanSet:
        def __init__(self, base, name):
            self._base = base
            self._name = name

        def __contains__(self, value):
            return value in self._base

        def _reject_scan(self, *args, **kwargs):
            raise AssertionError(
                f"sparse prepare enumerated accepted {self._name}")

        __iter__ = _reject_scan
        __len__ = _reject_scan

    # Keep the real accepted state, but turn every historical control/state
    # index into a point-read-only spy. This catches hidden dict copies and
    # scans that row-overlay access counts alone would miss.
    for name in (
            "_descriptors", "_selector_indexes", "_dependency_endpoint_index",
            "_dependency_endpoint_index_by_rank",
            "_unfinished_dependency_pins_by_segment", "_descriptor_state",
            "_unfinished_dependency_waiters_by_endpoint",
            "_cursor_runs_by_segment", "_cursor_versions_by_segment",
            "_active_transfers_by_run", "_active_transfer_id_counts",
            "_active_endpoint_counts", "_cursor_unissued_by_segment",
            "_source_release_watermarks", "_release_watermark_keys_by_domain",
            "_slots_by_key", "_target_ready_shards_by_slot",
            "_completed_terminal_intervals_by_rank", "_credits_by_key",
            "_obligations_by_id",
            "_slot_owner_counts", "_source_occupancy", "_target_occupancy",
            "_credit_ack_counts", "_ack_pins_by_segment",
            "_active_descriptor_count_by_domain", "_slot_row_counts_by_domain",
            "_credit_row_counts_by_domain", "_obligations_by_segment",
            "_obligation_remaining_by_segment"):
        value = getattr(ledger, name)
        if isinstance(value, dict):
            if name == "_cursor_runs_by_segment":
                value = {segment: NoScanMapping(rows, name + "[segment]")
                         for segment, rows in value.items()}
            setattr(ledger, name, NoScanMapping(value, name))
    ledger._active_segments = NoScanSet(
        ledger._active_segments, "_active_segments")

    visits = {"row_get": 0, "row_has": 0, "aggregate_get": 0}
    row_get = pipeline_ledger_module._RowOverlay.get
    row_has = pipeline_ledger_module._RowOverlay.has
    aggregate_get = pipeline_ledger_module._CountOverlay.get

    def counted_row_get(self, *args, **kwargs):
        visits["row_get"] += 1
        return row_get(self, *args, **kwargs)

    def counted_row_has(self, *args, **kwargs):
        visits["row_has"] += 1
        return row_has(self, *args, **kwargs)

    def counted_aggregate_get(self, *args, **kwargs):
        visits["aggregate_get"] += 1
        return aggregate_get(self, *args, **kwargs)

    with mock.patch.object(pipeline_ledger_module._RowOverlay,
                           "get", counted_row_get), \
            mock.patch.object(pipeline_ledger_module._RowOverlay,
                              "has", counted_row_has), \
            mock.patch.object(pipeline_ledger_module._CountOverlay,
                              "get", counted_aggregate_get):
        prepared = ledger.prepare_envelope(envelope)
    visits["row_accesses"] = sum(visits.values())
    visits["encoded_bytes"] = len(prepared.parsed.canonical_bytes)
    visits["header_normalized_payload_bytes"] = (
        _header_normalized_payload_bytes(envelope))
    return visits


def _node_event(seq, kind, work_id, payload=None):
    return {
        "logical_seq": seq,
        "tick": 0,
        "kind": kind,
        "work_id": work_id,
        "payload": payload or {},
    }


def _noc_resource(remaining_bytes, as_of_tick):
    request_id, stage, generation, rank, flow_id = "noc", "", 0, 0, 0
    node_id = flow_id
    node_work_id = canonical_work_id(
        request_id, stage, generation, rank, node_id)
    return {
        "resource_type": "noc_flow",
        "work_id": leg_work_id(node_work_id, f"noc:{flow_id}"),
        "request_id": request_id,
        "stage": stage,
        "generation": generation,
        "rank": rank,
        "node_id": node_id,
        "operation": "COMM_READ",
        "leg": f"noc:{flow_id}",
        "state": "Active",
        "total": {"ops": None, "bytes": 100, "latency_ns": None},
        "remaining": {"ops": None, "bytes": remaining_bytes, "latency_ns": None},
        "progress_source": "authoritative",
        "service_unit": "bytes",
        "route": [0, 1],
        "as_of_tick": as_of_tick,
        "issued_tick": 0,
        "descriptor_segment_id": None,
        "endpoint_selector": None,
        "flow_id": flow_id,
        "current_rate_Bpns": 2.5,
        "last_rate_update_time": float(as_of_tick),
    }


class PipelineLedgerV2Test(unittest.TestCase):
    def test_v4_claim_rejects_empty_descriptor_authority(self):
        wire = copy.deepcopy(_descriptor_wire())
        # Preserve the group axis maximum with a separate rank-2 shard, while
        # rank 1 owns only a one-tile short-tail shard. Tile 2 is group-valid
        # but has no active target membership for the descriptor owner.
        wire["groups"][0]["shards"] = [
            {
                "shard_ordinal": 0, "source_rank": 0, "target_rank": 2,
                "total_data_bytes": 5, "bytes_per_unit": 1,
                "noc_path": [0, 2],
            },
            {
                "shard_ordinal": 1, "source_rank": 0, "target_rank": 1,
                "total_data_bytes": 1, "bytes_per_unit": 1,
                "noc_path": [0, 1],
            },
        ]
        wire["logical_id_ranges"] = []
        wire["tag_allocation_runs"][0]["count"] = 6
        wire["tag_allocation_runs"][0]["shard_count"] = 2
        for run in wire["endpoint_runs"]:
            run["tile_count"] = 1
            run["shard_ordinals"] = [1]
            for leg in run["legs"]:
                leg["tag_offset"] = 1
        for run in wire["endpoint_dependency_runs"]:
            run["tile_count"] = 1
            run["shard_ordinals"] = [1]
        for run in wire["consume_completion_runs"]:
            run["tile_count"] = 1
            run["shard_ordinals"] = [1]
        for anchor in wire["anchors"]:
            selector = anchor.get("selector", {})
            if "shard_ordinal" in selector:
                selector["shard_ordinal"] = 1
                selector["tag_offset"] = 1
        descriptor = PipelineDescriptor.from_wire(wire)
        descriptor, full = _v4_full_with_descriptor(descriptor)
        ledger = PipelineLedgerV4()
        ledger.apply_envelope(full)
        claim = _pipeline_run(
            descriptor.identity.segment_id, 2, "consume_done", 1)
        owner = {
            "segment_id": descriptor.identity.segment_id,
            "pipeline_generation": descriptor.identity.generation,
            "stage_epoch": descriptor.identity.stage_epoch,
            "staging_allocation_id": descriptor.identity.staging_allocation_id,
            "group_ordinal": 0, "tile_ordinal": 2,
        }
        target_key = (descriptor.identity.buffer_domain_id, "target",
                      descriptor.identity.rank, None, 1)
        claim["slot_patches"] = [_v4_target_delta(
            target_key, 0, 1, 0, [{
                "op": "claim", "slot_generation": 1,
                "owner": owner, "target_shards": [],
            }])]
        frame = _use_v4_identity(_empty_envelope(
            "delta", 1, base_watermark=1, base_version=1, tick=1))
        frame.update({
            "logical_seq_start": 2, "logical_seq_end": 2,
            "logical_event_count": 1, "result_state_version": 2,
            "pipeline_transition_runs": [claim],
        })
        before = ledger.snapshot_state()
        with self.assertRaisesRegex(
                PipelineLedgerError, "exact descriptor membership"):
            ledger.prepare_envelope(frame)
        self.assertIs(ledger.snapshot_state(), before)
        self.assertFalse(ledger._slots_by_key[
            target_key]["occupied"])

    def test_v4_commit_rejects_archive_prepared_update_before_publication(self):
        archive_update = PipelineLedger().prepare_envelope(_empty_envelope(
            "full", 0, base_watermark=0, base_version=0))
        live = PipelineLedgerV4()
        before = live.snapshot_state()
        with self.assertRaisesRegex(PipelineLedgerError, "protocol"):
            live.commit_prepared(archive_update)
        self.assertIs(live.snapshot_state(), before)
        self.assertEqual((live.delivery_sequence, live.watermark,
                          live.state_version), (-1, 0, 0))

    def test_online_bridge_requires_v4_child_and_handle(self):
        scheduler = object.__new__(LiveOnlineSchedulerBase)
        outer = {
            "schema_version": 2, "delivery_sequence": 0,
            "delivery_epoch": 0, "tick": 0, "deferred_from_tick": None,
            "reasons": [], "arrivals": [], "completed_groups": [],
            "completed_nodes": [], "affected_ranks": [],
            "snapshot_handle": {"kind": "work_ledger_v3",
                                "epoch": 0, "tick": 0},
            "work_ledger": {"schema_version": 3},
        }
        with self.assertRaisesRegex(ValueError, "schema_version must be protocol 4"):
            scheduler._validate_schema(outer)
        outer["work_ledger"] = {"schema_version": 4}
        outer["snapshot_handle"]["kind"] = "work_ledger_v4"
        scheduler._validate_schema(outer)
        pipeline = PipelineLedgerV4()
        work_ledger = WorkLedger()
        live_frame = _use_v4_identity(_empty_envelope(
            "full", 0, base_watermark=0, base_version=0))
        parsed = pipeline.validate_envelope(live_frame)
        tx = work_ledger.begin_protocol_v3_transaction(
            delivery_sequence=live_frame.get("delivery_sequence"),
            tick=live_frame.get("tick"),
            snapshot_kind=live_frame.get("snapshot_kind"))
        pipeline.commit_prepared(pipeline.prepare_envelope(
            live_frame, work_transaction=tx, _parsed=parsed))
        tx.commit(notify=False)
        tx.notify()
        self.assertEqual(pipeline.delivery_sequence, 0)

    def test_v4_target_delta_claims_exact_membership_and_preserves_v3_archive(self):
        descriptor = _descriptor_with_shard_count(8)
        descriptor, full = _v4_full_with_descriptor(descriptor)
        ledger = PipelineLedgerV4()
        ledger.apply_envelope(full)
        retained_snapshot = ledger.snapshot_state()
        initialized_keys = {
            (descriptor.identity.buffer_domain_id, "target",
             descriptor.identity.rank, None, slot)
            for slot in (0, 1)
        }
        self.assertEqual(
            initialized_keys,
            {PipelineLedger._slot_key(row)
             for row in retained_snapshot["slots"]})

        issue = _v4_issue_delta(
            descriptor, delivery=1, watermark=1, issue_count=2)
        bad = copy.deepcopy(issue)
        target_delta = next(
            patch for transition in bad["pipeline_transition_runs"]
            for patch in transition["slot_patches"]
            if isinstance(patch, dict) and patch.get("form") == "target_delta")
        claim = target_delta["operations"][0]
        self.assertGreater(len(claim["target_shards"]), 1)
        claim["target_shards"].pop()
        with self.assertRaisesRegex(
                PipelineLedgerError, "exact descriptor membership"):
            ledger.prepare_envelope(bad)
        self.assertIs(ledger.snapshot_state(), retained_snapshot)
        self.assertEqual((ledger.delivery_sequence, ledger.watermark,
                          ledger.state_version), (0, 1, 1))

        ledger.apply_envelope(issue)
        target_rows = [row for row in ledger.snapshot_state()["slots"]
                       if row["role"] == "target" and row["occupied"]]
        self.assertEqual(len(target_rows), 1)
        owner = target_rows[0]["owner"]
        expected_targets = tuple(
            shard_ordinal for shard_ordinal in range(8)
            if (lambda shard: shard.emit_endpoint
                and shard.target_rank == descriptor.identity.rank)(
                    descriptor.selected_shard_tile(
                        owner["group_ordinal"], owner["tile_ordinal"] - 1,
                        shard_ordinal)[1]))
        self.assertEqual(target_rows[0]["target_shards_to_consume"],
                         expected_targets)
        key = PipelineLedger._slot_key(target_rows[0])
        self.assertEqual(ledger._target_slot_lineage[key],
                         (1, issue["logical_seq_end"], 0))

        consume_run_index = next(
            index for index, run in enumerate(descriptor.consume_completion_runs)
            if (run.group_ordinal == owner["group_ordinal"]
                and run.first_tile <= owner["tile_ordinal"]
                < run.first_tile + run.tile_count
                and target_rows[0]["rank"] == run.rank
                and target_rows[0]["target_shards_to_consume"][0]
                    in run.shard_ordinals))
        completion = descriptor.consume_completion_runs[consume_run_index]
        shard_position = completion.shard_ordinals.index(
            target_rows[0]["target_shards_to_consume"][0])
        terminal_store_id = (
            completion.first_dependency_json_id
            + (owner["tile_ordinal"] - completion.first_tile)
                * completion.dependency_tile_stride
            + shard_position * completion.dependency_shard_stride + 1)
        ledger._completed_terminal_intervals_by_rank[completion.rank] = (
            pipeline_ledger_module._completed_interval_add(
                None, terminal_store_id))
        wrong_consume = _pipeline_run(
            descriptor.identity.segment_id, 4, "consume_done", 2,
            cursor_start=issue["pipeline_transition_runs"][-1][
                "cursor_version_end"],
            cursor_end=issue["pipeline_transition_runs"][-1][
                "cursor_version_end"])
        wrong_consume["slot_patches"] = [_v4_target_delta(
            key, 1, issue["logical_seq_end"], 0, [{
                "op": "consume", "slot_generation": 1,
                "cause": {
                    "terminal_rank": completion.rank,
                    "terminal_store_id": terminal_store_id,
                    "consume_run_index": consume_run_index,
                },
                "add_shards": [(1 << 64) - 1], "cleared": True,
            }])]
        wrong_consume_frame = _empty_envelope(
            "delta", 2, base_watermark=issue["logical_seq_end"],
            base_version=2, tick=2)
        wrong_consume_frame = _use_v4_identity(wrong_consume_frame)
        wrong_consume_frame.update({
            "logical_seq_start": issue["logical_seq_end"] + 1,
            "logical_seq_end": issue["logical_seq_end"] + 1,
            "logical_event_count": 1,
            "result_state_version": 3,
            "pipeline_transition_runs": [wrong_consume],
        })
        before_bad_consume = ledger.snapshot_state()
        with self.assertRaisesRegex(
                PipelineLedgerError, "exact cause mapping"):
            ledger.prepare_envelope(wrong_consume_frame)
        self.assertIs(ledger.snapshot_state(), before_bad_consume)

        with self.assertRaises(PipelineLedgerError):
            PipelineLedger().validate_envelope(full)
        with self.assertRaises(PipelineLedgerError):
            PipelineLedgerV4().validate_envelope(_empty_envelope(
                "full", 0, base_watermark=0, base_version=0))

    def test_v4_consume_cause_clears_after_ready_proof_and_replays_immutably(self):
        descriptor = PipelineDescriptor.from_wire(_descriptor_wire())
        descriptor, full = _v4_full_with_descriptor(descriptor)
        ledger = PipelineLedgerV4()
        ledger.apply_envelope(full)
        issue = _v4_issue_delta(
            descriptor, delivery=1, watermark=1, issue_count=2)
        ledger.apply_envelope(issue)
        run_index = next(
            index for index, run in enumerate(descriptor.endpoint_runs)
            if run.rank == descriptor.identity.rank
            and run.legs[0].endpoint == "data_recv")
        endpoint_done_sequence = ledger.watermark + 1
        endpoint_done = _v4_endpoint_done_frame(
            ledger, descriptor, run_index,
            delivery=ledger.delivery_sequence + 1,
            watermark=ledger.watermark, base_version=ledger.state_version,
            sequence=endpoint_done_sequence,
            cursor_start=ledger._cursor_versions_by_segment[
                descriptor.identity.segment_id], tick=2)
        data_ready_sequence = endpoint_done_sequence + 1
        data_ready = _v4_data_ready_frame(
            descriptor, run_index,
            delivery=ledger.delivery_sequence + 1,
            watermark=ledger.watermark, base_version=ledger.state_version,
            sequence=data_ready_sequence,
            cursor_version=endpoint_done["pipeline_transition_runs"][0][
                "cursor_version_end"], tick=2)
        # data_ready touches the same cursor run as endpoint_done. Keep both
        # sequenced transitions in one delivery so the final cursor patch is
        # the changed endpoint_done row, not an unchanged republish.
        endpoint_done["logical_seq_end"] = data_ready_sequence
        endpoint_done["logical_event_count"] += 1
        endpoint_done["pipeline_transition_runs"].append(
            data_ready["pipeline_transition_runs"][0])
        ledger.apply_envelope(endpoint_done)
        old_snapshot = ledger.snapshot_state()
        old_target_row = next(
            row for row in old_snapshot["slots"]
            if row["role"] == "target" and row["occupied"])
        owner = old_target_row["owner"]
        key = PipelineLedger._slot_key(old_target_row)
        shard = descriptor.endpoint_runs[run_index].shard_ordinals[0]
        owner_tag = pipeline_ledger_module._target_owner_tag(owner)
        ready_key = (key[0], key[2], key[4], shard)
        self.assertEqual(
            ledger._target_ready_shards_by_slot[ready_key], owner_tag)
        completion = descriptor.consume_completion_runs[0]
        terminal_store_id = completion.first_dependency_json_id + 1
        ledger._completed_terminal_intervals_by_rank[completion.rank] = (
            pipeline_ledger_module._completed_interval_add(
                None, terminal_store_id))
        consume = _v4_consume_frame(
            descriptor, delivery=ledger.delivery_sequence + 1,
            watermark=ledger.watermark, base_version=ledger.state_version,
            predecessor_seq=issue["logical_seq_end"],
            predecessor_count=0, slot_generation=1, target_key=key,
            cursor_version=ledger._cursor_versions_by_segment[
                descriptor.identity.segment_id],
            operations=[{
                "op": "consume", "slot_generation": 1,
                "cause": {
                    "terminal_rank": completion.rank,
                    "terminal_store_id": terminal_store_id,
                    "consume_run_index": 0,
                },
                "add_shards": [shard], "cleared": True,
            }])
        receipt = ledger.apply_envelope(consume)
        self.assertFalse(receipt.replayed)
        current_snapshot = ledger.snapshot_state()
        current_target = next(
            row for row in current_snapshot["slots"]
            if PipelineLedger._slot_key(row) == key)
        self.assertFalse(current_target["occupied"])
        self.assertEqual(current_target["consumed_target_shards"], ())
        self.assertNotIn(ready_key, ledger._target_ready_shards_by_slot)
        self.assertEqual(ledger._target_slot_lineage[key],
                         (1, consume["logical_seq_end"], 0))
        self.assertTrue(old_target_row["occupied"])
        self.assertEqual(old_target_row["consumed_target_shards"], ())

        replay = ledger.apply_envelope(copy.deepcopy(consume))
        self.assertTrue(replay.replayed)
        self.assertIs(ledger.snapshot_state(), current_snapshot)

    def test_v4_same_patch_applies_shared_zero_stride_runs_in_wire_order(self):
        descriptor = _v4_partitioned_consume_descriptor(
            3, shared_terminal=True)
        descriptor, full = _v4_full_with_descriptor(descriptor)
        ledger = PipelineLedgerV4()
        ledger.apply_envelope(full)
        issue = _v4_issue_delta(
            descriptor, delivery=1, watermark=1, issue_count=2)
        ledger.apply_envelope(issue)
        target_row = next(
            row for row in ledger.snapshot_state()["slots"]
            if row["role"] == "target" and row["occupied"])
        key = PipelineLedger._slot_key(target_row)
        terminal_store_id = (
            descriptor.consume_completion_runs[0].first_dependency_json_id + 1)
        ledger._completed_terminal_intervals_by_rank[1] = (
            pipeline_ledger_module._completed_interval_add(
                None, terminal_store_id))
        consume = _v4_consume_frame(
            descriptor, delivery=2, watermark=issue["logical_seq_end"],
            base_version=2, predecessor_seq=issue["logical_seq_end"],
            predecessor_count=0, slot_generation=1, target_key=key,
            operations=[
                {
                    "op": "consume", "slot_generation": 1,
                    "cause": {"terminal_rank": 1,
                             "terminal_store_id": terminal_store_id,
                             "consume_run_index": 0},
                    "add_shards": [0], "cleared": False,
                },
                {
                    "op": "consume", "slot_generation": 1,
                    "cause": {"terminal_rank": 1,
                             "terminal_store_id": terminal_store_id,
                             "consume_run_index": 1},
                    "add_shards": [1], "cleared": False,
                },
            ])
        ledger.apply_envelope(consume)
        row = next(item for item in ledger.snapshot_state()["slots"]
                   if PipelineLedger._slot_key(item) == key)
        self.assertTrue(row["occupied"])
        self.assertEqual(row["target_shards_to_consume"], (0, 1, 2))
        self.assertEqual(row["consumed_target_shards"], (0, 1))
        self.assertEqual(ledger._target_slot_lineage[key],
                         (1, consume["logical_seq_end"], 2))

    def test_v4_bad_target_delta_after_node_completion_is_dual_ledger_atomic(self):
        descriptor = PipelineDescriptor.from_wire(_descriptor_wire())
        completion = descriptor.consume_completion_runs[0]
        terminal_store_id = completion.first_dependency_json_id + 1
        terminal_work_id = canonical_work_id(
            descriptor.identity.request_id, descriptor.identity.stage,
            descriptor.identity.generation, completion.rank,
            terminal_store_id)
        terminal_payload = {
            "request_id": descriptor.identity.request_id,
            "stage": descriptor.identity.stage,
            "generation": descriptor.identity.generation,
            "rank": completion.rank, "node_id": terminal_store_id,
            "operation": "compute",
            "total": {"ops": 1, "bytes": 0, "latency_ns": 1},
            "dependencies": [],
        }
        descriptor, full = _full_with_descriptor(
            descriptor=descriptor,
            node_events=(_node_event(
                2, "commit", terminal_work_id, terminal_payload),))
        full = _use_v4_identity(full)
        for slot_index in (0, 1):
            key = (descriptor.identity.buffer_domain_id, "target",
                   descriptor.identity.rank, None, slot_index)
            full["pipeline_transition_runs"][0]["slot_patches"].append(
                _v4_target_delta(
                    key, 0, 0, 0,
                    [{"op": "initialize", "slot_generation": 0}]))
        pipeline = PipelineLedgerV4()
        work = WorkLedger()

        def _apply(frame):
            # Reference-decoder application sequence (the production entry
            # routes to the native dual-ledger core since the v4 round);
            # mirrors online_scheduler_base._apply_work_ledger's original
            # validate -> begin -> prepare -> work commit -> pipeline commit
            # ordering 1:1.
            parsed_frame = pipeline.validate_envelope(frame)
            tx = work.begin_protocol_v3_transaction(
                delivery_sequence=frame.get("delivery_sequence"),
                tick=frame.get("tick"),
                snapshot_kind=frame.get("snapshot_kind"))
            prepared = pipeline.prepare_envelope(
                frame, work_transaction=tx, _parsed=parsed_frame)
            tx.commit(notify=False)
            pipeline.commit_prepared(prepared)
            tx.notify()

        _apply(full)
        issue = _v4_issue_delta(
            descriptor, delivery=1, watermark=2, issue_count=2)
        _apply(issue)
        before_pipeline = pipeline.snapshot_state()
        before_work = work.snapshot_state()
        before_lineage = dict(pipeline._target_slot_lineage)
        target_row = next(
            row for row in before_pipeline["slots"]
            if row["role"] == "target" and row["occupied"])
        target_key = PipelineLedger._slot_key(target_row)

        completion_event = _node_event(
            5, "completion", terminal_work_id, terminal_payload)
        completion_event["tick"] = 2
        consume = _pipeline_run(
            descriptor.identity.segment_id, 6, "consume_done", 2,
            cursor_start=2, cursor_end=2)
        consume["slot_patches"] = [_v4_target_delta(
            target_key, 1, issue["logical_seq_end"], 0, [{
                "op": "consume", "slot_generation": 1,
                "cause": {
                    "terminal_rank": completion.rank,
                    "terminal_store_id": terminal_store_id,
                    "consume_run_index": 0,
                },
                "add_shards": [(1 << 64) - 1], "cleared": True,
            }])]
        bad = _use_v4_identity(_empty_envelope(
            "delta", 2, base_watermark=issue["logical_seq_end"],
            base_version=2, tick=2))
        bad.update({
            "logical_seq_start": 5, "logical_seq_end": 6,
            "logical_event_count": 2, "result_state_version": 3,
            "node_transitions": [completion_event],
            "pipeline_transition_runs": [consume],
        })
        with self.assertRaisesRegex(
                PipelineLedgerError, "exact cause mapping"):
            _apply(bad)
        self.assertIs(pipeline.snapshot_state(), before_pipeline)
        self.assertIs(work.snapshot_state(), before_work)
        self.assertEqual((pipeline.delivery_sequence, pipeline.watermark,
                          pipeline.state_version), (1, 4, 2))
        self.assertEqual((work.last_delivery_sequence,
                          work.event_watermark), (1, 4))
        self.assertEqual(pipeline._target_slot_lineage, before_lineage)
        self.assertTrue(target_row["occupied"])

    def test_v4_target_claim_erase_reinitialize_reclaim_keeps_tombstone_lineage(self):
        old_descriptor = PipelineDescriptor.from_wire(_descriptor_wire())
        old_descriptor, full = _v4_full_with_descriptor(old_descriptor)
        ledger = PipelineLedgerV4()
        ledger.apply_envelope(full)
        target_key = (old_descriptor.identity.buffer_domain_id, "target",
                      old_descriptor.identity.rank, None, 0)

        issue, issued_key, target_run_index, active = _v4_single_target_issue(
            old_descriptor, delivery=1, watermark=1, base_version=1,
            sequence=2, cursor_start=0, slot_generation=0,
            predecessor_seq=1)
        self.assertEqual(issued_key, target_key)
        ledger.apply_envelope(issue)
        self.assertEqual(ledger._target_slot_lineage[target_key], (1, 2, 0))

        close = _v4_request_close_frame(
            ledger, old_descriptor, delivery=2, watermark=2,
            base_version=2, sequence=3, cursor_start=1,
            target_key=target_key)
        ledger.apply_envelope(close)
        self.assertEqual(ledger._target_slot_lineage[target_key], (1, 3, 0))

        endpoint_done = _v4_closed_endpoint_done_frame(
            ledger, old_descriptor, target_run_index, delivery=3,
            watermark=3, base_version=3, sequence=4,
            cursor_start=2, active=active)
        ledger.apply_envelope(endpoint_done)

        settle = _pipeline_run(
            old_descriptor.identity.segment_id, 5, "settle_done", 1,
            cursor_start=3, cursor_end=4)
        settle_frame = _use_v4_identity(_empty_envelope(
            "delta", 4, base_watermark=4, base_version=4, tick=1))
        settle_frame.update({
            "logical_seq_start": 5, "logical_seq_end": 5,
            "logical_event_count": 1, "result_state_version": 5,
            "pipeline_transition_runs": [settle],
            "cursor_patches": [{
                "segment_id": old_descriptor.identity.segment_id,
                "cursor_version": 4, "retired": False, "runs": [],
            }],
        })
        ledger.apply_envelope(settle_frame)

        new_descriptor = _v4_descriptor_same_domain_next_segment(
            old_descriptor, ordinal=1, shift=100)
        _new_descriptor, new_full = _full_with_descriptor(
            descriptor=new_descriptor)
        retire = _pipeline_run(
            old_descriptor.identity.segment_id, 6, "descriptor_retire", 1,
            cursor_start=4, cursor_end=5)
        retire_patches = []
        initialize_patches = []
        for slot_index in (0, 1):
            key = (old_descriptor.identity.buffer_domain_id, "target",
                   old_descriptor.identity.rank, None, slot_index)
            generation, predecessor, count = ledger._target_slot_lineage[key]
            retire_patches.append(_v4_target_delta(
                key, generation, predecessor, count, [{
                    "op": "erase", "slot_generation": generation,
                }]))
            initialize_patches.append(_v4_target_delta(
                key, generation, 6, 0, [{
                    "op": "initialize", "slot_generation": generation,
                }]))
        retire["slot_patches"] = retire_patches
        new_commit = _pipeline_run(
            new_descriptor.identity.segment_id, 7, "descriptor_commit", 1)
        new_commit["obligation_patches"] = copy.deepcopy(
            new_full["pipeline_transition_runs"][0]["obligation_patches"])
        new_commit["slot_patches"] = initialize_patches
        new_cursor = {
            "segment_id": new_descriptor.identity.segment_id,
            "cursor_version": 0, "retired": False,
            "runs": copy.deepcopy(new_full["cursor_patches"][0]["runs"]),
        }
        retired_cursor = {
            "segment_id": old_descriptor.identity.segment_id,
            "cursor_version": 5, "retired": True,
            "runs": [{"run_index": index, "value": None}
                     for index in range(len(old_descriptor.endpoint_runs))],
        }
        lifecycle = _use_v4_identity(_empty_envelope(
            "delta", 5, base_watermark=5, base_version=5, tick=1))
        lifecycle.update({
            "logical_seq_start": 6, "logical_seq_end": 7,
            "logical_event_count": 2, "result_state_version": 6,
            "descriptor_commits": [new_full["descriptor_commits"][0]],
            "pipeline_transition_runs": [retire, new_commit],
            "cursor_patches": sorted(
                [retired_cursor, new_cursor],
                key=lambda item: item["segment_id"]),
        })
        ledger.apply_envelope(lifecycle)
        self.assertEqual(ledger._target_slot_lineage[target_key], (1, 7, 0))
        self.assertFalse(ledger._slots_by_key[target_key]["occupied"])

        reclaim = _v4_issue_delta(
            new_descriptor, delivery=6, watermark=7, issue_count=2)
        reclaim["base_state_version"] = 6
        reclaim["result_state_version"] = 7
        for transition in reclaim["pipeline_transition_runs"]:
            for delta in transition["slot_patches"]:
                if isinstance(delta, dict) and delta.get("form") == "target_delta":
                    delta["before_slot_generation"] = 1
                    delta["predecessor_logical_seq"] = 7
                    delta["predecessor_consumed_count"] = 0
                    delta["operations"][0]["slot_generation"] = 2
        ledger.apply_envelope(reclaim)
        reclaimed = ledger._slots_by_key[target_key]
        self.assertTrue(reclaimed["occupied"])
        self.assertEqual(reclaimed["owner"]["segment_id"],
                         new_descriptor.identity.segment_id)
        self.assertEqual(ledger._target_slot_lineage[target_key], (2, 9, 0))

    def test_v4_stale_owner_callback_after_reclaim_is_rejected(self):
        # Clone the legal close->erase->initialize->reclaim lifecycle, then
        # deliver a stale consume that still names the retired owner's
        # generation and predecessor lineage. The reclaimed slot must stay
        # untouched by the old callback.
        old_descriptor = PipelineDescriptor.from_wire(_descriptor_wire())
        old_descriptor, full = _v4_full_with_descriptor(old_descriptor)
        ledger = PipelineLedgerV4()
        ledger.apply_envelope(full)
        target_key = (old_descriptor.identity.buffer_domain_id, "target",
                      old_descriptor.identity.rank, None, 0)
        issue, issued_key, target_run_index, active = _v4_single_target_issue(
            old_descriptor, delivery=1, watermark=1, base_version=1,
            sequence=2, cursor_start=0, slot_generation=0,
            predecessor_seq=1)
        ledger.apply_envelope(issue)
        close = _v4_request_close_frame(
            ledger, old_descriptor, delivery=2, watermark=2,
            base_version=2, sequence=3, cursor_start=1,
            target_key=target_key)
        ledger.apply_envelope(close)
        endpoint_done = _v4_closed_endpoint_done_frame(
            ledger, old_descriptor, target_run_index, delivery=3,
            watermark=3, base_version=3, sequence=4,
            cursor_start=2, active=active)
        ledger.apply_envelope(endpoint_done)
        settle = _pipeline_run(
            old_descriptor.identity.segment_id, 5, "settle_done", 1,
            cursor_start=3, cursor_end=4)
        settle_frame = _use_v4_identity(_empty_envelope(
            "delta", 4, base_watermark=4, base_version=4, tick=1))
        settle_frame.update({
            "logical_seq_start": 5, "logical_seq_end": 5,
            "logical_event_count": 1, "result_state_version": 5,
            "pipeline_transition_runs": [settle],
            "cursor_patches": [{
                "segment_id": old_descriptor.identity.segment_id,
                "cursor_version": 4, "retired": False, "runs": [],
            }],
        })
        ledger.apply_envelope(settle_frame)
        new_descriptor = _v4_descriptor_same_domain_next_segment(
            old_descriptor, ordinal=1, shift=100)
        _new_descriptor, new_full = _full_with_descriptor(
            descriptor=new_descriptor)
        retire = _pipeline_run(
            old_descriptor.identity.segment_id, 6, "descriptor_retire", 1,
            cursor_start=4, cursor_end=5)
        retire_patches = []
        initialize_patches = []
        for slot_index in (0, 1):
            key = (old_descriptor.identity.buffer_domain_id, "target",
                   old_descriptor.identity.rank, None, slot_index)
            generation, predecessor, count = ledger._target_slot_lineage[key]
            retire_patches.append(_v4_target_delta(
                key, generation, predecessor, count, [{
                    "op": "erase", "slot_generation": generation,
                }]))
            initialize_patches.append(_v4_target_delta(
                key, generation, 6, 0, [{
                    "op": "initialize", "slot_generation": generation,
                }]))
        retire["slot_patches"] = retire_patches
        new_commit = _pipeline_run(
            new_descriptor.identity.segment_id, 7, "descriptor_commit", 1)
        new_commit["obligation_patches"] = copy.deepcopy(
            new_full["pipeline_transition_runs"][0]["obligation_patches"])
        new_commit["slot_patches"] = initialize_patches
        new_cursor = {
            "segment_id": new_descriptor.identity.segment_id,
            "cursor_version": 0, "retired": False,
            "runs": copy.deepcopy(new_full["cursor_patches"][0]["runs"]),
        }
        retired_cursor = {
            "segment_id": old_descriptor.identity.segment_id,
            "cursor_version": 5, "retired": True,
            "runs": [{"run_index": index, "value": None}
                     for index in range(len(old_descriptor.endpoint_runs))],
        }
        lifecycle = _use_v4_identity(_empty_envelope(
            "delta", 5, base_watermark=5, base_version=5, tick=1))
        lifecycle.update({
            "logical_seq_start": 6, "logical_seq_end": 7,
            "logical_event_count": 2, "result_state_version": 6,
            "descriptor_commits": [new_full["descriptor_commits"][0]],
            "pipeline_transition_runs": [retire, new_commit],
            "cursor_patches": sorted(
                [retired_cursor, new_cursor],
                key=lambda item: item["segment_id"]),
        })
        ledger.apply_envelope(lifecycle)
        reclaim = _v4_issue_delta(
            new_descriptor, delivery=6, watermark=7, issue_count=2)
        reclaim["base_state_version"] = 6
        reclaim["result_state_version"] = 7
        for transition in reclaim["pipeline_transition_runs"]:
            for delta in transition["slot_patches"]:
                if isinstance(delta, dict) and delta.get("form") == "target_delta":
                    delta["before_slot_generation"] = 1
                    delta["predecessor_logical_seq"] = 7
                    delta["predecessor_consumed_count"] = 0
                    delta["operations"][0]["slot_generation"] = 2
        ledger.apply_envelope(reclaim)
        self.assertEqual(ledger._target_slot_lineage[target_key], (2, 9, 0))
        before = ledger.snapshot_state()

        completion = old_descriptor.consume_completion_runs[0]
        stale = _v4_consume_frame(
            old_descriptor, delivery=7, watermark=ledger.watermark,
            base_version=ledger.state_version,
            predecessor_seq=7, predecessor_count=0, slot_generation=1,
            target_key=target_key, cursor_version=0,
            operations=[{
                "op": "consume", "slot_generation": 1,
                "cause": {
                    "terminal_rank": completion.rank,
                    "terminal_store_id": (
                        completion.first_dependency_json_id + 1),
                    "consume_run_index": 0,
                },
                "add_shards": [0], "cleared": True,
            }])
        with self.assertRaisesRegex(
                PipelineLedgerError,
                "retired descriptor received a later transition"):
            ledger.apply_envelope(stale)
        self.assertIs(ledger.snapshot_state(), before)
        self.assertEqual(ledger._target_slot_lineage[target_key], (2, 9, 0))
        reclaimed_row = ledger._slots_by_key[target_key]
        self.assertTrue(reclaimed_row["occupied"])
        self.assertEqual(reclaimed_row["owner"]["segment_id"],
                         new_descriptor.identity.segment_id)
        self.assertEqual(reclaimed_row["consumed_target_shards"], ())

    def test_v4_cause_generation_and_predecessor_errors_are_atomic(self):
        descriptor = _descriptor_with_shard_count(2)
        descriptor, full = _v4_full_with_descriptor(descriptor)
        ledger = PipelineLedgerV4()
        ledger.apply_envelope(full)
        issue = _v4_issue_delta(
            descriptor, delivery=1, watermark=1, issue_count=2)
        ledger.apply_envelope(issue)
        before = ledger.snapshot_state()
        key = PipelineLedger._slot_key(next(
            row for row in before["slots"]
            if row["role"] == "target" and row["occupied"]))
        completion = descriptor.consume_completion_runs[0]
        terminal_root = None
        for shard_position, _shard in enumerate(completion.shard_ordinals):
            terminal_root = pipeline_ledger_module._completed_interval_add(
                terminal_root,
                completion.first_dependency_json_id + 1 + shard_position)
        ledger._completed_terminal_intervals_by_rank[completion.rank] = (
            terminal_root)

        def bad_frame(*, predecessor=None, operation_generation=1,
                      run_index=0, store_offset=0,
                      cause_run_index=None, add_shards=(0,)):
            completion = descriptor.consume_completion_runs[run_index]
            store_id = completion.first_dependency_json_id + 1 + store_offset
            transition = _pipeline_run(
                descriptor.identity.segment_id, 4, "consume_done", 2,
                cursor_start=2, cursor_end=2)
            transition["slot_patches"] = [_v4_target_delta(
                key, 1, issue["logical_seq_end"] if predecessor is None
                else predecessor, 0, [{
                    "op": "consume",
                    "slot_generation": operation_generation,
                    "cause": {
                        "terminal_rank": completion.rank,
                        "terminal_store_id": store_id,
                        "consume_run_index": (
                            run_index if cause_run_index is None
                            else cause_run_index),
                    },
                    "add_shards": list(add_shards), "cleared": False,
                }])]
            frame = _use_v4_identity(_empty_envelope(
                "delta", 2, base_watermark=3, base_version=2, tick=2))
            frame.update({
                "logical_seq_start": 4, "logical_seq_end": 4,
                "logical_event_count": 1, "result_state_version": 3,
                "pipeline_transition_runs": [transition],
            })
            return frame

        bad_cases = (
            (bad_frame(predecessor=2), "predecessor lineage"),
            (bad_frame(operation_generation=2), "occupied owner row"),
            (bad_frame(cause_run_index=99), "outside the descriptor"),
            (bad_frame(store_offset=2), "outside its run"),
            # Both invalid rows preserve the required one-member count. The
            # first names shard 1 for shard 0's marker; the second names
            # shard 0 for shard 1's valid affine marker.
            (bad_frame(add_shards=(1,)), "exact cause mapping"),
            (bad_frame(store_offset=1), "exact cause mapping"),
        )
        for frame, error in bad_cases:
            with self.subTest(error=error):
                with self.assertRaisesRegex(PipelineLedgerError, error):
                    ledger.prepare_envelope(frame)
                self.assertIs(ledger.snapshot_state(), before)
                self.assertEqual(ledger._target_slot_lineage[key],
                                 (1, issue["logical_seq_end"], 0))

    def test_v4_early_clear_without_ready_and_terminal_proof_is_rejected(self):
        descriptor = PipelineDescriptor.from_wire(_descriptor_wire())
        descriptor, full = _v4_full_with_descriptor(descriptor)
        ledger = PipelineLedgerV4()
        ledger.apply_envelope(full)
        issue = _v4_issue_delta(
            descriptor, delivery=1, watermark=1, issue_count=2)
        ledger.apply_envelope(issue)
        # Seed the completed-terminal interval so the consume cause mapping
        # itself is legal; the missing data_recv endpoint completion is the
        # only absent element of the clear proof.
        completion_seed = descriptor.consume_completion_runs[0]
        ledger._completed_terminal_intervals_by_rank[completion_seed.rank] = (
            pipeline_ledger_module._completed_interval_add(
                None, completion_seed.first_dependency_json_id + 1))
        # No endpoint_done/data_ready deliveries: the occupied owner row
        # exists, but data_recv completion is not proven for any shard.
        before = ledger.snapshot_state()
        before_lineage = dict(ledger._target_slot_lineage)
        target_row = next(
            row for row in before["slots"]
            if row["role"] == "target" and row["occupied"])
        key = PipelineLedger._slot_key(target_row)
        owner_shards = list(target_row["target_shards_to_consume"])
        self.assertTrue(owner_shards)
        completion = descriptor.consume_completion_runs[0]
        terminal_store_id = completion.first_dependency_json_id + 1
        consume = _v4_consume_frame(
            descriptor, delivery=2, watermark=issue["logical_seq_end"],
            base_version=2, predecessor_seq=issue["logical_seq_end"],
            predecessor_count=0, slot_generation=1, target_key=key,
            operations=[{
                "op": "consume", "slot_generation": 1,
                "cause": {
                    "terminal_rank": completion.rank,
                    "terminal_store_id": terminal_store_id,
                    "consume_run_index": 0,
                },
                "add_shards": owner_shards, "cleared": True,
            }])
        with self.assertRaisesRegex(
                PipelineLedgerError,
                "cleared before its ready/terminal proof"):
            ledger.apply_envelope(consume)
        self.assertIs(ledger.snapshot_state(), before)
        self.assertEqual(ledger._target_slot_lineage, before_lineage)
        self.assertEqual(
            (ledger.delivery_sequence, ledger.watermark,
             ledger.state_version), (1, issue["logical_seq_end"], 2))
        still = next(
            row for row in ledger.snapshot_state()["slots"]
            if PipelineLedger._slot_key(row) == key)
        self.assertTrue(still["occupied"])
        self.assertEqual(still["consumed_target_shards"], ())

    def test_v4_nonzero_initial_full_is_explicitly_rejected(self):
        descriptor = PipelineDescriptor.from_wire(_descriptor_wire())
        _unused, full = _v4_full_with_descriptor(descriptor)

        late = copy.deepcopy(full)
        late["delivery_sequence"] = 1
        late["logical_seq_start"] = 1
        late["logical_seq_end"] = 1
        late["replay_token"] = "cpp_online/test-run/1"
        # A non-zero delivery on a fresh ledger is rejected by the explicit
        # sequence gate before the full-snapshot gate is reached.
        with self.assertRaisesRegex(
                PipelineLedgerError,
                "delivery_sequence gap or stale replay"):
            PipelineLedgerV4().apply_envelope(late)

        based = copy.deepcopy(full)
        based["base_state_version"] = 1
        with self.assertRaisesRegex(
                PipelineLedgerError,
                "full snapshot is only valid for initial state"):
            PipelineLedgerV4().apply_envelope(based)

        ledger = PipelineLedgerV4()
        first = ledger.apply_envelope(copy.deepcopy(full))
        self.assertFalse(first.replayed)
        # The byte-identical duplicate full with the same replay token is the
        # specified idempotent replay, not a second initialization.
        duplicate = ledger.apply_envelope(copy.deepcopy(full))
        self.assertTrue(duplicate.replayed)
        # A later full frame that claims to reinitialize the run state is
        # explicitly rejected instead of being accepted as a resync.
        refull = _use_v4_identity(_empty_envelope(
            "full", 1, base_watermark=0, base_version=0))
        refull["logical_seq_start"] = 1
        refull["logical_seq_end"] = 0
        with self.assertRaisesRegex(
                PipelineLedgerError,
                "full snapshot is only valid for initial state"):
            ledger.apply_envelope(refull)

    def test_inactive_target_tail_cannot_enter_consume_completion_runs(self):
        wire = _descriptor_with_shard_count(2).to_wire()
        wire["groups"][0]["shards"][1]["total_data_bytes"] = 1
        with self.assertRaisesRegex(
                pipeline_descriptor_module.PipelineDescriptorError,
                "cover every active target tile"):
            PipelineDescriptor.from_wire(wire)

    def test_v4_lineage_survives_clear_erase_reinitialize_and_reclaim(self):
        old_descriptor = PipelineDescriptor.from_wire(_descriptor_wire())
        old_descriptor, full = _v4_full_with_descriptor(old_descriptor)
        ledger = PipelineLedgerV4()
        ledger.apply_envelope(full)
        target_key = (old_descriptor.identity.buffer_domain_id, "target",
                      old_descriptor.identity.rank, None, 0)
        issue, _key, target_run_index, active = _v4_single_target_issue(
            old_descriptor, delivery=1, watermark=1, base_version=1,
            sequence=2, cursor_start=0, slot_generation=0,
            predecessor_seq=1)
        ledger.apply_envelope(issue)
        self.assertEqual(ledger._target_slot_lineage[target_key], (1, 2, 0))

        close = _v4_request_close_frame(
            ledger, old_descriptor, delivery=2, watermark=2,
            base_version=2, sequence=3, cursor_start=1,
            target_key=target_key)
        ledger.apply_envelope(close)
        self.assertEqual(ledger._target_slot_lineage[target_key], (1, 3, 0))

        endpoint_done = _v4_closed_endpoint_done_frame(
            ledger, old_descriptor, target_run_index, delivery=3,
            watermark=3, base_version=3, sequence=4,
            cursor_start=2, active=active)
        ledger.apply_envelope(endpoint_done)

        settle = _pipeline_run(
            old_descriptor.identity.segment_id, 5, "settle_done", 1,
            cursor_start=3, cursor_end=4)
        settle_frame = _use_v4_identity(_empty_envelope(
            "delta", 4, base_watermark=4, base_version=4, tick=1))
        settle_frame.update({
            "logical_seq_start": 5, "logical_seq_end": 5,
            "logical_event_count": 1, "result_state_version": 5,
            "pipeline_transition_runs": [settle],
            "cursor_patches": [{
                "segment_id": old_descriptor.identity.segment_id,
                "cursor_version": 4, "retired": False, "runs": [],
            }],
        })
        ledger.apply_envelope(settle_frame)

        new_descriptor = _v4_descriptor_same_domain_next_segment(
            old_descriptor, ordinal=1, shift=100)
        _new_descriptor, new_full = _full_with_descriptor(
            descriptor=new_descriptor)
        retire = _pipeline_run(
            old_descriptor.identity.segment_id, 6, "descriptor_retire", 1,
            cursor_start=4, cursor_end=5)
        retire_patches = []
        for slot_index in (0, 1):
            key = (old_descriptor.identity.buffer_domain_id, "target",
                   old_descriptor.identity.rank, None, slot_index)
            generation, predecessor, count = ledger._target_slot_lineage[key]
            retire_patches.append(_v4_target_delta(
                key, generation, predecessor, count, [{
                    "op": "erase", "slot_generation": generation,
                }]))
        retire["slot_patches"] = retire_patches
        new_commit = _pipeline_run(
            new_descriptor.identity.segment_id, 7, "descriptor_commit", 1)
        new_commit["obligation_patches"] = copy.deepcopy(
            new_full["pipeline_transition_runs"][0]["obligation_patches"])
        new_init_patches = []
        for slot_index in (0, 1):
            key = (new_descriptor.identity.buffer_domain_id, "target",
                   new_descriptor.identity.rank, None, slot_index)
            generation, _previous_seq, count = ledger._target_slot_lineage[key]
            new_init_patches.append(_v4_target_delta(
                key, generation, 6, count, [{
                    "op": "initialize", "slot_generation": generation,
                }]))
        new_commit["slot_patches"] = new_init_patches
        new_full_cursor = copy.deepcopy(new_full["cursor_patches"][0])
        new_cursor = {
            "segment_id": new_descriptor.identity.segment_id,
            "cursor_version": 0, "retired": False,
            "runs": new_full_cursor["runs"],
        }
        retire_cursor = {
            "segment_id": old_descriptor.identity.segment_id,
            "cursor_version": 5, "retired": True,
            "runs": [{"run_index": index, "value": None}
                     for index in range(len(old_descriptor.endpoint_runs))],
        }
        lifecycle = _use_v4_identity(_empty_envelope(
            "delta", 5, base_watermark=5, base_version=5, tick=1))
        lifecycle.update({
            "logical_seq_start": 6, "logical_seq_end": 7,
            "logical_event_count": 2, "result_state_version": 6,
            "descriptor_commits": [new_full["descriptor_commits"][0]],
            "pipeline_transition_runs": [retire, new_commit],
            "cursor_patches": sorted(
                [retire_cursor, new_cursor],
                key=lambda row: row["segment_id"]),
        })
        ledger.apply_envelope(lifecycle)
        self.assertEqual(ledger._target_slot_lineage[target_key], (1, 7, 0))
        self.assertFalse(ledger._slots_by_key[target_key]["occupied"])

        reclaim = _v4_issue_delta(
            new_descriptor, delivery=6, watermark=7, issue_count=2)
        reclaim["base_state_version"] = 6
        reclaim["result_state_version"] = 7
        for transition in reclaim["pipeline_transition_runs"]:
            for delta in transition["slot_patches"]:
                if isinstance(delta, dict) and delta.get("form") == "target_delta":
                    delta["before_slot_generation"] = 1
                    delta["predecessor_logical_seq"] = 7
                    delta["predecessor_consumed_count"] = 0
                    delta["operations"][0]["slot_generation"] = 2
        ledger.apply_envelope(reclaim)
        reclaimed = ledger._slots_by_key[target_key]
        self.assertTrue(reclaimed["occupied"])
        self.assertEqual(reclaimed["owner"]["segment_id"],
                         new_descriptor.identity.segment_id)
        self.assertEqual(ledger._target_slot_lineage[target_key], (2, 9, 0))

    def test_v4_target_delta_cost_scales_with_declaration_and_actual_additions(self):
        touched_shards = 4
        for shard_count in (8, 32, 128):
            descriptor = _v4_partitioned_consume_descriptor(
                shard_count, shared_terminal=False)
            descriptor, full = _v4_full_with_descriptor(descriptor)
            ledger = PipelineLedgerV4()
            ledger.apply_envelope(full)
            issue = _v4_issue_delta(
                descriptor, delivery=1, watermark=1, issue_count=2)

            target_declaration_uint_reads = 0
            original_uint = pipeline_ledger_module._uint

            def counted_uint(value, path, maximum=pipeline_ledger_module.U64_MAX):
                nonlocal target_declaration_uint_reads
                if path.endswith(".target_shards[]"):
                    target_declaration_uint_reads += 1
                return original_uint(value, path, maximum)

            with mock.patch.object(
                    pipeline_ledger_module, "_uint", new=counted_uint):
                parsed_issue = ledger.validate_envelope(issue)
            self.assertEqual(target_declaration_uint_reads, shard_count)

            selected_geometry_reads = []
            original_selected = PipelineDescriptor.selected_shard_tile

            def counted_selected(self, group_ordinal, local_tile, shard):
                selected_geometry_reads.append(
                    (group_ordinal, local_tile, shard))
                return original_selected(self, group_ordinal, local_tile, shard)

            with mock.patch.object(
                    PipelineDescriptor, "selected_shard_tile",
                    new=counted_selected):
                prepared_issue = ledger.prepare_envelope(
                    issue, _parsed=parsed_issue)
            # The selected endpoint/cursor facts use shard 0. Every other
            # shard's membership geometry is touched exactly once by the
            # descriptor-backed claim proof.
            for shard in range(1, shard_count):
                self.assertEqual(
                    selected_geometry_reads.count((0, 0, shard)), 1)
            ledger.commit_prepared(prepared_issue)

            target_row = next(
                row for row in ledger.snapshot_state()["slots"]
                if row["role"] == "target" and row["occupied"])
            key = PipelineLedger._slot_key(target_row)
            completion_runs = descriptor.consume_completion_runs
            terminal_root = None
            operations = []
            for shard in range(touched_shards):
                run_index = shard
                completion = completion_runs[run_index]
                terminal_store_id = completion.first_dependency_json_id + 1
                terminal_root = pipeline_ledger_module._completed_interval_add(
                    terminal_root, terminal_store_id)
                operations.append({
                    "op": "consume", "slot_generation": 1,
                    "cause": {
                        "terminal_rank": completion.rank,
                        "terminal_store_id": terminal_store_id,
                        "consume_run_index": run_index,
                    },
                    "add_shards": [shard], "cleared": False,
                })
            ledger._completed_terminal_intervals_by_rank[1] = terminal_root
            consume = _v4_consume_runs_frame(
                descriptor, delivery=2, watermark=issue["logical_seq_end"],
                base_version=2, first_sequence=issue["logical_seq_end"] + 1,
                predecessor_seq=issue["logical_seq_end"],
                predecessor_count=0, slot_generation=1, target_key=key,
                cursor_version=2, operations=operations)

            actual_addition_uint_reads = 0

            def counted_delta_uint(
                    value, path, maximum=pipeline_ledger_module.U64_MAX):
                nonlocal actual_addition_uint_reads
                if path.endswith(".add_shards[]"):
                    actual_addition_uint_reads += 1
                return original_uint(value, path, maximum)

            with mock.patch.object(
                    pipeline_ledger_module, "_uint",
                    new=counted_delta_uint):
                parsed_consume = ledger.validate_envelope(consume)
            self.assertEqual(actual_addition_uint_reads, touched_shards)

            copies = []
            freezes = []
            original_copy = pipeline_ledger_module._copy_consumed_slot_members
            original_freeze = pipeline_ledger_module._freeze_consumed_slot_members

            def counted_copy(values):
                copies.append(len(values))
                return original_copy(values)

            def counted_freeze(values):
                freezes.append(len(values))
                return original_freeze(values)

            run_reads = {"indexed": [], "iterated": 0}
            descriptor_with_counted_runs = replace(
                descriptor,
                consume_completion_runs=_CountingTuple(
                    completion_runs, run_reads))
            ledger._descriptors[descriptor.identity.segment_id] = (
                descriptor_with_counted_runs)
            with mock.patch.object(
                    pipeline_ledger_module,
                    "_copy_consumed_slot_members", new=counted_copy), \
                    mock.patch.object(
                    pipeline_ledger_module,
                    "_freeze_consumed_slot_members", new=counted_freeze):
                prepared_consume = ledger.prepare_envelope(
                    consume, _parsed=parsed_consume)
            self.assertEqual(copies, [0])
            self.assertEqual(freezes, [touched_shards])
            self.assertEqual(run_reads["iterated"], 0)
            self.assertEqual(len(run_reads["indexed"]), 2 * touched_shards)
            self.assertEqual(
                run_reads["indexed"],
                [index for shard in range(touched_shards)
                 for index in (shard, shard)])
            ledger.commit_prepared(prepared_consume)
            result = next(
                row for row in ledger.snapshot_state()["slots"]
                if PipelineLedger._slot_key(row) == key)
            self.assertEqual(result["target_shards_to_consume"],
                             tuple(range(shard_count)))
            self.assertEqual(result["consumed_target_shards"],
                             tuple(range(touched_shards)))
            self.assertEqual(ledger._target_slot_lineage[key],
                             (1, consume["logical_seq_end"], touched_shards))
            print(
                "V4_TARGET_COST scope=seeded_consumer_prefix "
                f"S={shard_count} T={touched_shards} "
                f"claim_uint_reads={target_declaration_uint_reads} "
                f"geometry_calls={len(selected_geometry_reads)} "
                f"addition_uint_reads={actual_addition_uint_reads} "
                f"cause_tuple_index_reads={len(run_reads['indexed'])} "
                f"cause_tuple_rows_iterated={run_reads['iterated']} "
                f"cow_calls={len(copies)} cow_items={sum(copies)} "
                f"freeze_calls={len(freezes)} freeze_items={sum(freezes)} "
                f"target_array_elems={len(result['target_shards_to_consume'])} "
                f"consumed_array_elems={len(result['consumed_target_shards'])} "
                f"encoded_bytes={len(parsed_consume.canonical_bytes)}")

    def test_v4_target_delta_fixed_owner_cost_scales_to_full_clear(self):
        shard_count = 128
        for transition_count in (8, 32, 128):
            with self.subTest(transition_count=transition_count):
                descriptor = _v4_all_target_endpoint_shards_descriptor(
                    shard_count)
                descriptor, full = _v4_full_with_descriptor(descriptor)
                ledger = PipelineLedgerV4()
                ledger.apply_envelope(full)

                issue, target_key, target_run_index, _active = (
                    _v4_single_target_issue(
                        descriptor, delivery=1, watermark=1, base_version=1,
                        sequence=2, cursor_start=0, slot_generation=0,
                        predecessor_seq=1))
                declared_member_reads = 0
                original_uint = pipeline_ledger_module._uint

                def counted_issue_uint(
                        value, path,
                        maximum=pipeline_ledger_module.U64_MAX):
                    nonlocal declared_member_reads
                    if path.endswith(".target_shards[]"):
                        declared_member_reads += 1
                    return original_uint(value, path, maximum)

                with mock.patch.object(
                        pipeline_ledger_module, "_uint",
                        new=counted_issue_uint):
                    parsed_issue = ledger.validate_envelope(issue)
                self.assertEqual(declared_member_reads, shard_count)

                geometry_reads = []
                original_selected = PipelineDescriptor.selected_shard_tile

                def counted_selected(self, group_ordinal, local_tile, shard):
                    geometry_reads.append((group_ordinal, local_tile, shard))
                    return original_selected(
                        self, group_ordinal, local_tile, shard)

                with mock.patch.object(
                        PipelineDescriptor, "selected_shard_tile",
                        new=counted_selected):
                    prepared_issue = ledger.prepare_envelope(
                        issue, _parsed=parsed_issue)
                for shard in range(1, shard_count):
                    self.assertEqual(
                        geometry_reads.count((0, 0, shard)),
                        2 if shard == 1 else 1)
                self.assertEqual(
                    len(geometry_reads), shard_count + 1,
                    "claim checks each declared shard once; cursor advance "
                    "also resolves the next active shard")
                ledger.commit_prepared(prepared_issue)

                _v4_seed_completed_target_owner(
                    ledger, descriptor, target_key, target_run_index)
                initial_snapshot = ledger.snapshot_state()
                initial_target = next(
                    row for row in initial_snapshot["slots"]
                    if PipelineLedger._slot_key(row) == target_key)
                self.assertTrue(initial_target["occupied"])
                self.assertEqual(initial_target["target_shards_to_consume"],
                                 tuple(range(shard_count)))
                self.assertEqual(initial_target["consumed_target_shards"], ())

                completion = descriptor.consume_completion_runs[0]
                operations = []
                for shard in range(transition_count):
                    terminal_store_id = (
                        completion.first_dependency_json_id
                        + shard * completion.dependency_shard_stride + 1)
                    operations.append({
                        "op": "consume", "slot_generation": 1,
                        "cause": {
                            "terminal_rank": completion.rank,
                            "terminal_store_id": terminal_store_id,
                            "consume_run_index": 0,
                        },
                        "add_shards": [shard],
                        "cleared": (
                            shard_count == transition_count
                            and shard == transition_count - 1),
                    })
                consume = _v4_consume_runs_frame(
                    descriptor, delivery=2,
                    watermark=issue["logical_seq_end"], base_version=2,
                    first_sequence=issue["logical_seq_end"] + 1,
                    predecessor_seq=issue["logical_seq_end"],
                    predecessor_count=0, slot_generation=1,
                    target_key=target_key,
                    cursor_version=ledger._cursor_versions_by_segment[
                        descriptor.identity.segment_id],
                    operations=operations)

                actual_addition_reads = 0

                def counted_consume_uint(
                        value, path,
                        maximum=pipeline_ledger_module.U64_MAX):
                    nonlocal actual_addition_reads
                    if path.endswith(".add_shards[]"):
                        actual_addition_reads += 1
                    return original_uint(value, path, maximum)

                with mock.patch.object(
                        pipeline_ledger_module, "_uint",
                        new=counted_consume_uint):
                    parsed_consume = ledger.validate_envelope(consume)
                self.assertEqual(actual_addition_reads, transition_count)

                run_reads = {"indexed": [], "iterated": 0}
                counted_descriptor = replace(
                    descriptor,
                    consume_completion_runs=_CountingTuple(
                        descriptor.consume_completion_runs, run_reads))
                ledger._descriptors[descriptor.identity.segment_id] = (
                    counted_descriptor)
                copies = []
                freezes = []
                original_copy = pipeline_ledger_module._copy_consumed_slot_members
                original_freeze = (
                    pipeline_ledger_module._freeze_consumed_slot_members)

                def counted_copy(values):
                    copies.append(len(values))
                    return original_copy(values)

                def counted_freeze(values):
                    freezes.append(len(values))
                    return original_freeze(values)

                with mock.patch.object(
                        pipeline_ledger_module,
                        "_copy_consumed_slot_members", new=counted_copy), \
                        mock.patch.object(
                        pipeline_ledger_module,
                        "_freeze_consumed_slot_members", new=counted_freeze):
                    prepared_consume = ledger.prepare_envelope(
                        consume, _parsed=parsed_consume)

                self.assertEqual(copies, [0])
                self.assertEqual(
                    freezes, [0 if transition_count == shard_count
                              else transition_count])
                self.assertEqual(run_reads["iterated"], 0)
                expected_run_reads = (2 * transition_count
                                      + (shard_count
                                         if transition_count == shard_count
                                         else 0))
                self.assertEqual(
                    run_reads["indexed"], [0] * expected_run_reads)

                ledger.commit_prepared(prepared_consume)
                final_snapshot = ledger.snapshot_state()
                final_target = next(
                    row for row in final_snapshot["slots"]
                    if PipelineLedger._slot_key(row) == target_key)
                expected_target = dict(initial_target)
                if transition_count == shard_count:
                    expected_target.update({
                        "occupied": False, "owner": None,
                        "consumed": False,
                        "target_shards_to_consume": (),
                        "consumed_target_shards": (),
                    })
                    expected_count = 0
                else:
                    expected_target["consumed"] = True
                    expected_target["consumed_target_shards"] = tuple(
                        range(transition_count))
                    expected_count = transition_count
                self.assertEqual(dict(final_target), expected_target)
                self.assertEqual(
                    ledger._target_slot_lineage[target_key],
                    (1, consume["logical_seq_end"], expected_count))
                self.assertTrue(initial_target["occupied"])
                self.assertEqual(initial_target["consumed_target_shards"], ())
                print(
                    "V4_TARGET_COST scope=seeded_accepted_consumer_state "
                    f"S={shard_count} T={transition_count} "
                    f"claim_uint_reads={declared_member_reads} "
                    f"geometry_calls={len(geometry_reads)} "
                    f"addition_uint_reads={actual_addition_reads} "
                    f"cause_tuple_index_reads={len(run_reads['indexed'])} "
                    f"cause_tuple_rows_iterated={run_reads['iterated']} "
                    f"cow_calls={len(copies)} cow_items={sum(copies)} "
                    f"freeze_calls={len(freezes)} freeze_items={sum(freezes)} "
                    f"target_array_elems={len(initial_target['target_shards_to_consume'])} "
                    f"consumed_array_elems={len(final_target['consumed_target_shards'])} "
                    f"encoded_bytes={len(parsed_consume.canonical_bytes)}")

    def test_completed_terminal_provenance_merges_adjacent_ids_persistently(self):
        root = None
        root = pipeline_ledger_module._completed_interval_add(root, 10)
        retained_prefix = root
        root = pipeline_ledger_module._completed_interval_add(root, 12)
        root = pipeline_ledger_module._completed_interval_add(root, 11)
        root = pipeline_ledger_module._completed_interval_add(root, 9)
        root = pipeline_ledger_module._completed_interval_add(root, 13)
        root = pipeline_ledger_module._completed_interval_add(root, 12)

        self.assertEqual((root.first, root.last), (9, 13))
        self.assertIsNone(root.left)
        self.assertIsNone(root.right)
        self.assertEqual((retained_prefix.first, retained_prefix.last),
                         (10, 10))
        self.assertFalse(
            pipeline_ledger_module._completed_interval_contains(
                retained_prefix, 12))

    def test_endpoint_done_and_descriptor_level_consume_done_bindings(self):
        descriptor = PipelineDescriptor.from_wire(_descriptor_wire())
        run = descriptor.endpoint_runs[0]
        leg = run.legs[0]
        endpoint_id = descriptor.endpoint_id(
            0, run.first_tile, run.shard_ordinals[0], 0)
        endpoint = _pipeline_run(
            descriptor.identity.segment_id, 1, "endpoint_done", 1)
        endpoint.update({
            "tile_first": run.first_tile,
            "tile_stride": 0,
            "rank_first": run.rank,
            "rank_stride": 0,
            "shard_first": run.shard_ordinals[0],
            "shard_stride": 0,
            "endpoint_selector": {
                "group_ordinal": run.group_ordinal,
                "tile_ordinal": run.first_tile,
                "shard_ordinal": run.shard_ordinals[0],
                "endpoint": leg.endpoint,
            },
            "store_id_first": endpoint_id + 1,
            "store_id_stride": 0,
            "legacy_context": {
                "request_id": run.legacy_context.request_id,
                "stage": run.legacy_context.stage,
                "generation": run.legacy_context.generation,
            },
            "resource_leg": "endpoint",
        })
        endpoint_amount = pipeline_ledger_module._expected_endpoint_amount(
            descriptor, 0, 0)
        endpoint.update({
            "resource_quantity_prefix": [
                {key: 0 if value is not None else None
                 for key, value in endpoint_amount.items()},
                endpoint_amount,
            ],
        })
        parsed = _parse_pipeline_runs(
            [endpoint], {descriptor.identity.segment_id: descriptor},
            "pipeline_transition_runs")
        self.assertEqual(parsed[0]["transition_kind"], "endpoint_done")

        ordinary_consume = _pipeline_run(
            descriptor.identity.segment_id, 2, "consume_done", 1)
        parsed = _parse_pipeline_runs(
            [ordinary_consume], {descriptor.identity.segment_id: descriptor},
            "pipeline_transition_runs")
        self.assertIsNone(parsed[0]["endpoint_selector"])

        invalid = copy.deepcopy(endpoint)
        invalid["endpoint_selector"] = None
        invalid["store_id_first"] = None
        invalid["store_id_stride"] = None
        invalid["legacy_context"] = None
        invalid["tile_first"] = invalid["tile_stride"] = None
        invalid["rank_first"] = invalid["rank_stride"] = None
        invalid["shard_first"] = invalid["shard_stride"] = None
        invalid["resource_leg"] = None
        empty_amount = {"ops": None, "bytes": None, "latency_ns": None}
        invalid["resource_quantity_prefix"] = [
            copy.deepcopy(empty_amount), copy.deepcopy(empty_amount)]
        with self.assertRaisesRegex(PipelineLedgerError, "concrete selector"):
            _parse_pipeline_runs(
                [invalid], {descriptor.identity.segment_id: descriptor},
                "pipeline_transition_runs")

    def test_parsed_pipeline_run_dto_keeps_only_prepare_fields(self):
        descriptor = PipelineDescriptor.from_wire(_descriptor_wire())
        segment = descriptor.identity.segment_id
        descriptor_level = _pipeline_run(
            segment, 1, "consume_done", 1)
        issue = _issue_delta(
            descriptor, delivery=1, watermark=1, issue_count=1
        )["pipeline_transition_runs"][0]
        endpoint_rows = []
        endpoint_kinds = (
            "tile_issue", "data_ready", "fill_ack_progress",
            "release_ack_progress", "endpoint_done",
        )
        for offset, kind in enumerate(endpoint_kinds, start=2):
            row = copy.deepcopy(issue)
            row["first_logical_seq"] = offset
            row["pattern_id"] = f"dto-probe/{kind}/{offset}"
            row["transition_kind"] = kind
            endpoint_rows.append(row)

        expected_keys = {
            "first_logical_seq", "descriptor_segment_id", "transition_kind",
            "endpoint_selector", "endpoint_run_index", "endpoint_leg_index",
            "endpoint_ordinal", "_selected_global_tile_ordinal",
            "_selected_shard_tile", "endpoint_store_id", "slot_patches",
            "credit_patches", "obligation_patches", "tick_first",
            "cursor_version_start", "cursor_version_end",
        }
        original_uint = pipeline_ledger_module._uint
        with mock.patch.object(
                pipeline_ledger_module, "_uint",
                wraps=original_uint) as uint_calls:
            parsed = _parse_pipeline_runs(
                [descriptor_level, *endpoint_rows], {segment: descriptor},
                "pipeline_transition_runs")

        self.assertEqual(len(parsed), len(endpoint_rows) + 1)
        self.assertEqual(set(parsed[0]), expected_keys)
        self.assertIsNone(parsed[0]["endpoint_selector"])
        self.assertIsNone(parsed[0]["endpoint_store_id"])
        self.assertEqual(parsed[0]["transition_kind"], "consume_done")
        store_id_reads = [
            call for call in uint_calls.call_args_list
            if len(call.args) > 1
            and call.args[1].endswith(".store_id_first")
        ]
        self.assertEqual(len(store_id_reads), len(endpoint_rows))
        for wire, dto in zip(endpoint_rows, parsed[1:]):
            self.assertEqual(set(dto), expected_keys)
            self.assertEqual(dto["first_logical_seq"],
                             wire["first_logical_seq"])
            self.assertEqual(dto["transition_kind"], wire["transition_kind"])
            self.assertEqual(dto["endpoint_selector"],
                             wire["endpoint_selector"])
            self.assertEqual(dto["endpoint_store_id"],
                             wire["store_id_first"])
            self.assertIsInstance(dto["slot_patches"], tuple)
            self.assertIsInstance(dto["credit_patches"], tuple)
            self.assertIsInstance(dto["obligation_patches"], tuple)
            self.assertNotIn("legacy_context", dto)
            self.assertNotIn("resource_quantity_prefix", dto)

    def test_pipeline_run_dto_removes_measured_wire_mapping_copies(self):
        descriptor = PipelineDescriptor.from_wire(_descriptor_wire())
        segment = descriptor.identity.segment_id
        row = _issue_delta(
            descriptor, delivery=1, watermark=1, issue_count=1
        )["pipeline_transition_runs"][0]

        def counted_row(visits):
            values = copy.deepcopy(row)
            values["endpoint_selector"] = _CountedWireMapping(
                values["endpoint_selector"], "selector", visits)
            values["legacy_context"] = _CountedWireMapping(
                values["legacy_context"], "legacy", visits)
            values["resource_quantity_prefix"] = [
                _CountedWireMapping(amount, f"quantity{index}", visits)
                for index, amount in enumerate(
                    values["resource_quantity_prefix"])
            ]
            return _CountedWireMapping(values, "outer", visits)

        visits = {name: 0 for name in
                  ("outer", "selector", "legacy", "quantity0", "quantity1")}
        parsed = _parse_pipeline_runs(
            [counted_row(visits)], {segment: descriptor},
            "pipeline_transition_runs")[0]
        self.assertEqual(visits, {
            "outer": 0, "selector": 1, "legacy": 0,
            "quantity0": 0, "quantity1": 0,
        })
        self.assertEqual(len(parsed), 16)

        # Recreate the old retention path independently on fresh counted wire
        # mappings. ``dict(mapping)`` invokes the mapping's real ``keys`` path.
        old_visits = {name: 0 for name in visits}
        old_wire = counted_row(old_visits)
        old_dto = dict(old_wire)
        selector = old_wire["endpoint_selector"]
        old_dto["endpoint_selector"] = dict(selector)
        old_dto["endpoint_run_index"] = parsed["endpoint_run_index"]
        old_dto["endpoint_leg_index"] = parsed["endpoint_leg_index"]
        old_dto["endpoint_ordinal"] = parsed["endpoint_ordinal"]
        old_dto["_selected_global_tile_ordinal"] = (
            parsed["_selected_global_tile_ordinal"])
        old_dto["_selected_shard_tile"] = parsed["_selected_shard_tile"]
        old_dto["endpoint_store_id"] = parsed["endpoint_store_id"]
        old_dto["legacy_context"] = dict(old_wire["legacy_context"])
        old_dto["slot_patches"] = parsed["slot_patches"]
        old_dto["credit_patches"] = parsed["credit_patches"]
        old_dto["obligation_patches"] = parsed["obligation_patches"]
        old_dto["resource_quantity_prefix"] = tuple(
            dict(amount) for amount in old_wire["resource_quantity_prefix"])
        old_dto["tick_first"] = parsed["tick_first"]
        old_dto["cursor_version_start"] = parsed["cursor_version_start"]
        old_dto["cursor_version_end"] = parsed["cursor_version_end"]
        self.assertEqual(old_visits, {
            "outer": 1, "selector": 1, "legacy": 1,
            "quantity0": 1, "quantity1": 1,
        })
        self.assertEqual(len(old_dto), 30)
        print(
            "[pipeline-run-dto-copy-work] "
            f"old_keys={len(old_dto)} retained_keys={len(parsed)} "
            f"old_outer_legacy_quantity_dict_copies="
            f"{old_visits['outer'] + old_visits['legacy'] + old_visits['quantity0'] + old_visits['quantity1']} "
            f"new_outer_legacy_quantity_dict_copies="
            f"{visits['outer'] + visits['legacy'] + visits['quantity0'] + visits['quantity1']}")

    def test_pipeline_run_dto_scale_preserves_fields_and_canonical_bytes(self):
        expected_keys = {
            "first_logical_seq", "descriptor_segment_id", "transition_kind",
            "endpoint_selector", "endpoint_run_index", "endpoint_leg_index",
            "endpoint_ordinal", "_selected_global_tile_ordinal",
            "_selected_shard_tile", "endpoint_store_id", "slot_patches",
            "credit_patches", "obligation_patches", "tick_first",
            "cursor_version_start", "cursor_version_end",
        }
        for count in (8, 32, 128):
            with self.subTest(count=count):
                descriptor, envelope = _full_with_descriptor()
                segment = descriptor.identity.segment_id
                rows = [
                    _pipeline_run(segment, seq, "service_done", 1)
                    for seq in range(1, count + 1)
                ]
                envelope["pipeline_transition_runs"] = rows
                envelope["logical_seq_end"] = count
                envelope["logical_event_count"] = count
                expected_bytes = json.dumps(
                    envelope, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False, allow_nan=False).encode("utf-8")
                parsed = PipelineLedger().validate_envelope(envelope)
                self.assertEqual(parsed.canonical_bytes, expected_bytes)
                self.assertEqual(len(parsed.pipeline_runs), count)

                retained_keys = 0
                old_keys = 0
                for wire, actual in zip(
                        rows, parsed.pipeline_runs):
                    old = dict(wire)
                    old["endpoint_selector"] = None
                    old["endpoint_run_index"] = None
                    old["endpoint_leg_index"] = None
                    old["endpoint_ordinal"] = None
                    old["_selected_global_tile_ordinal"] = None
                    old["_selected_shard_tile"] = None
                    old["endpoint_store_id"] = None
                    old["legacy_context"] = None
                    old["slot_patches"] = ()
                    old["credit_patches"] = ()
                    old["obligation_patches"] = ()
                    old["resource_quantity_prefix"] = tuple(
                        dict(amount)
                        for amount in wire["resource_quantity_prefix"])
                    old["tick_first"] = wire["tick_first"]
                    old["cursor_version_start"] = (
                        wire["cursor_version_start"])
                    old["cursor_version_end"] = wire["cursor_version_end"]
                    self.assertEqual(set(actual), expected_keys)
                    self.assertEqual(
                        actual, {key: old[key] for key in expected_keys})
                    retained_keys += len(actual)
                    old_keys += len(old)
                self.assertEqual(retained_keys, sum(
                    len(row) for row in parsed.pipeline_runs))
                self.assertGreater(old_keys, retained_keys)
                print(
                    "[pipeline-run-dto-scale] "
                    f"T={count} old_retained_keys={old_keys} "
                    f"new_retained_keys={retained_keys} "
                    f"canonical_bytes={len(parsed.canonical_bytes)}")

    def test_pipeline_run_dto_invalid_fields_reject_atomically_and_replay_is_exact(self):
        descriptor, full = _full_with_descriptor()
        ledger = PipelineLedger()
        full_receipt = ledger.apply_envelope(full)
        issue = _issue_delta(
            descriptor, delivery=1, watermark=1, issue_count=1)
        issue_receipt = ledger.apply_envelope(issue)
        before_replay = ledger.snapshot_state()
        replay_receipt = ledger.apply_envelope(copy.deepcopy(issue))
        self.assertEqual(
            (full_receipt.snapshot_kind, full_receipt.delivery_sequence,
             full_receipt.logical_event_count, full_receipt.event_watermark,
             full_receipt.replayed), ("full", 0, 1, 1, False))
        self.assertEqual(
            (issue_receipt.snapshot_kind, issue_receipt.delivery_sequence,
             issue_receipt.logical_event_count, issue_receipt.event_watermark,
             issue_receipt.replayed), ("delta", 1, 1, 2, False))
        self.assertEqual(
            (replay_receipt.snapshot_kind, replay_receipt.delivery_sequence,
             replay_receipt.logical_event_count,
             replay_receipt.event_watermark, replay_receipt.replayed),
            ("delta", 1, 1, 2, True))
        self.assertIs(ledger.snapshot_state(), before_replay)

        def bad_legacy(row):
            row["legacy_context"]["stage"] += "/changed"

        def bad_stride(row):
            row["store_id_stride"] = 1

        def bool_store(row):
            row["store_id_first"] = True

        def overflow_store(row):
            row["store_id_first"] = pipeline_ledger_module.U64_MAX + 1

        def wrong_quantity(row):
            quantity = row["resource_quantity_prefix"][1]
            quantity["bytes"] = quantity["bytes"] + 1

        invalid_cases = (
            ("legacy", bad_legacy, "legacy_context disagrees"),
            ("stride", bad_stride, "store_id_stride must be 0"),
            ("bool store", bool_store, "store_id_first must be uint"),
            ("overflow store", overflow_store,
             "store_id_first must be uint"),
            ("quantity", wrong_quantity,
             "resource quantity does not match descriptor endpoint"),
        )
        for name, mutate, error in invalid_cases:
            with self.subTest(invalid=name):
                scheduler = object.__new__(OnlineSchedulerBase)
                scheduler.joint_pipeline_ledger = PipelineLedger()
                scheduler.joint_work_ledger = WorkLedger()
                scheduler.work_ledger_absent_seen = False
                scheduler._apply_work_ledger({"work_ledger": full})
                pipeline = scheduler.joint_pipeline_ledger
                work = scheduler.joint_work_ledger
                before_pipeline = pipeline.snapshot_state()
                before_work = work.snapshot_state()
                pipeline_marks = (
                    pipeline.delivery_sequence, pipeline.watermark,
                    pipeline.state_version)
                work_marks = (
                    work.last_delivery_sequence, work.event_watermark,
                    work.version)
                bad = _issue_delta(
                    descriptor, delivery=1, watermark=1, issue_count=1)
                mutate(bad["pipeline_transition_runs"][0])
                with mock.patch.object(
                        work, "begin_protocol_v3_transaction") as begin_txn:
                    with self.assertRaisesRegex(PipelineLedgerError, error):
                        scheduler._apply_work_ledger({"work_ledger": bad})
                begin_txn.assert_not_called()
                self.assertIs(pipeline.snapshot_state(), before_pipeline)
                self.assertIs(work.snapshot_state(), before_work)
                self.assertEqual(
                    (pipeline.delivery_sequence, pipeline.watermark,
                     pipeline.state_version), pipeline_marks)
                self.assertEqual(
                    (work.last_delivery_sequence, work.event_watermark,
                     work.version), work_marks)

    def test_obligation_postwrite_compare_guard_and_atomic_rejections(self):
        descriptor, full = _full_with_descriptor()
        ledger = PipelineLedger()
        ledger.apply_envelope(full)
        issue = _issue_delta(
            descriptor, delivery=1, watermark=1, issue_count=1)
        parsed = ledger.validate_envelope(issue)
        self.assertEqual(
            len(parsed.pipeline_runs[0]["obligation_patches"]), 1)

        owned_obligations = []
        compare_visits = {"new_row": 0}
        original_freeze = pipeline_ledger_module._freeze_dto
        original_equal = pipeline_ledger_module._typed_equal
        obligation_fields = {
            "obligation_id", "segment_id", "rank", "first_store_id",
            "remaining_endpoint_count", "state",
        }

        def counted_freeze(value):
            frozen = original_freeze(value)
            if (type(frozen) is MappingProxyType
                    and set(frozen) == obligation_fields):
                owned_obligations.append(frozen)
            return frozen

        def counted_equal(left, right):
            if any(left is row or right is row
                   for row in owned_obligations):
                compare_visits["new_row"] += 1
            return original_equal(left, right)

        with mock.patch.object(
                pipeline_ledger_module, "_freeze_dto", counted_freeze), \
                mock.patch.object(
                    pipeline_ledger_module, "_typed_equal", counted_equal):
            prepared = ledger.prepare_envelope(issue, _parsed=parsed)
        self.assertFalse(prepared.replayed)
        self.assertEqual(len(owned_obligations), 1)
        # The first new-vs-derived row equality proves the exact object later
        # returned by _RowOverlay.get; the builtin-only guard omits its second
        # recursive equality.
        self.assertEqual(compare_visits["new_row"], 1)

        def assert_atomic_rejection(mutate, error):
            scheduler = object.__new__(OnlineSchedulerBase)
            scheduler.joint_pipeline_ledger = PipelineLedger()
            scheduler.joint_work_ledger = WorkLedger()
            scheduler.work_ledger_absent_seen = False
            scheduler._apply_work_ledger({"work_ledger": full})
            pipeline = scheduler.joint_pipeline_ledger
            work = scheduler.joint_work_ledger
            before_pipeline = pipeline.snapshot_state()
            before_work = work.snapshot_state()
            pipeline_marks = (
                pipeline.delivery_sequence, pipeline.watermark,
                pipeline.state_version)
            work_marks = (
                work.last_delivery_sequence, work.event_watermark,
                work.version)
            bad = _issue_delta(
                descriptor, delivery=1, watermark=1, issue_count=1)
            mutate(bad["pipeline_transition_runs"][0])
            with self.assertRaisesRegex(PipelineLedgerError, error):
                scheduler._apply_work_ledger({"work_ledger": bad})
            self.assertIs(pipeline.snapshot_state(), before_pipeline)
            self.assertIs(work.snapshot_state(), before_work)
            self.assertEqual(
                (pipeline.delivery_sequence, pipeline.watermark,
                 pipeline.state_version), pipeline_marks)
            self.assertEqual(
                (work.last_delivery_sequence, work.event_watermark,
                 work.version), work_marks)

        def omit_changed_patch(row):
            row["obligation_patches"] = []

        def stale_before_row(row):
            row["obligation_patches"][0]["before"][
                "remaining_endpoint_count"] += 1

        def delete_expected_active_row(row):
            row["obligation_patches"][0]["after"] = None

        assert_atomic_rejection(
            omit_changed_patch,
            "obligation patch set does not match endpoint progress")
        assert_atomic_rejection(
            stale_before_row,
            "obligation patch before row differs from accepted COW state")
        assert_atomic_rejection(
            delete_expected_active_row,
            "obligation patch does not match endpoint completion/close progress")

        # _string intentionally accepts str subclasses. A toggle equality on
        # the after row passes the first comparison and fails the historical
        # post-write comparison; its old-before subtype also forces fallback.
        visits = {"after_expected_comparisons": 0}
        descriptor_wire = _descriptor_wire()
        descriptor_wire["identity"]["segment_id"] = _ToggleEqStr(
            descriptor_wire["identity"]["segment_id"], "descriptor", visits)
        custom_descriptor = PipelineDescriptor.from_wire(descriptor_wire)
        _custom_descriptor, custom_full = _full_with_descriptor(
            descriptor=custom_descriptor)
        scheduler = object.__new__(OnlineSchedulerBase)
        scheduler.joint_pipeline_ledger = PipelineLedger()
        scheduler.joint_work_ledger = WorkLedger()
        scheduler.work_ledger_absent_seen = False
        scheduler._apply_work_ledger({"work_ledger": custom_full})
        pipeline = scheduler.joint_pipeline_ledger
        work = scheduler.joint_work_ledger
        before_pipeline = pipeline.snapshot_state()
        before_work = work.snapshot_state()
        bad = _issue_delta(
            custom_descriptor, delivery=1, watermark=1, issue_count=1)
        patch = bad["pipeline_transition_runs"][0]["obligation_patches"][0]
        patch["before"]["segment_id"] = _ToggleEqStr(
            str(patch["before"]["segment_id"]), "before", visits)
        patch["after"]["segment_id"] = _ToggleEqStr(
            str(patch["after"]["segment_id"]), "after", visits)
        postwrite_pair_calls = {"count": 0}
        original_prepare = pipeline.prepare_envelope

        def reset_and_prepare(envelope, *, work_transaction=None,
                              _parsed=None):
            visits["after_expected_comparisons"] = 0
            postwrite_pair_calls["count"] = 0
            return original_prepare(
                envelope, work_transaction=work_transaction, _parsed=_parsed)

        def count_after_expected(left, right):
            if (type(left) is MappingProxyType and type(right) is dict
                    and isinstance(left.get("segment_id"), _ToggleEqStr)
                    and left["segment_id"].role == "after"
                    and isinstance(right.get("segment_id"), _ToggleEqStr)
                    and right["segment_id"].role == "descriptor"):
                postwrite_pair_calls["count"] += 1
            return original_equal(left, right)

        with mock.patch.object(
                pipeline, "prepare_envelope", reset_and_prepare), \
                mock.patch.object(
                    pipeline_ledger_module, "_typed_equal",
                    count_after_expected):
            with self.assertRaisesRegex(
                    PipelineLedgerError,
                    "obligation state is inconsistent with cursor progress"):
                scheduler._apply_work_ledger({"work_ledger": bad})
        self.assertEqual(postwrite_pair_calls["count"], 2)
        self.assertEqual(visits["after_expected_comparisons"], 2)
        self.assertIs(pipeline.snapshot_state(), before_pipeline)
        self.assertIs(work.snapshot_state(), before_work)

    def test_empty_full_delta_and_exact_replay(self):
        ledger = PipelineLedger()
        full = _empty_envelope(
            "full", 0, base_watermark=0, base_version=0)
        self.assertFalse(ledger.apply_envelope(full).replayed)
        self.assertTrue(ledger.apply_envelope(copy.deepcopy(full)).replayed)
        delta = _empty_envelope(
            "delta", 1, base_watermark=0, base_version=0, tick=1)
        self.assertFalse(ledger.apply_envelope(delta).replayed)
        state = ledger.snapshot_state()
        self.assertEqual(state["delivery_sequence"], 1)
        self.assertEqual(state["event_watermark"], 0)
        self.assertIs(state, ledger.snapshot_state())

    def test_group_global_slot_parity_and_atomic_rejection(self):
        descriptor = _two_group_physical_descriptor()
        segment = descriptor.identity.segment_id
        domain = descriptor.identity.buffer_domain_id

        def slot_row(role, tile, slot):
            source = role == "source"
            owner = {
                "segment_id": segment,
                "pipeline_generation": descriptor.identity.generation,
                "stage_epoch": descriptor.identity.stage_epoch,
                "staging_allocation_id": (
                    descriptor.identity.staging_allocation_id),
                "group_ordinal": 1,
                "tile_ordinal": tile,
            }
            return {
                "buffer_domain_id": domain,
                "role": role,
                "rank": 0 if source else descriptor.identity.rank,
                "shard_ordinal": 0 if source else None,
                "slot_index": slot,
                "occupied": True,
                "owner": owner,
                "consumed": False,
                "target_shards_to_consume": [] if source else [0],
                "consumed_target_shards": [],
            }

        # Group 1 starts at global tile 4, so local tiles 1 and 2 own slots
        # 1 and 0 respectively. Exercise the shared before/after row parser
        # against real source and target endpoint runs from PhysicalPipelineDraft.
        source_tile_1 = slot_row("source", 1, 1)
        source_tile_2 = slot_row("source", 2, 0)
        target_tile_1 = slot_row("target", 1, 1)
        target_tile_2 = slot_row("target", 2, 0)
        parse_run = _pipeline_run(segment, 1, "consume_done", 0)
        parse_run["slot_patches"] = [
            {"before": None, "after": source_tile_2},
            {"before": source_tile_1, "after": None},
            {"before": None, "after": target_tile_2},
            {"before": target_tile_1, "after": None},
        ]
        parsed = _parse_pipeline_runs(
            [parse_run], {segment: descriptor}, "pipeline_transition_runs")
        observed = {
            (row["role"], row["owner"]["tile_ordinal"], row["slot_index"])
            for _key, before, after in parsed[0]["slot_patches"]
            for row in (before, after) if row is not None
        }
        self.assertEqual(observed, {
            ("source", 1, 1), ("source", 2, 0),
            ("target", 1, 1), ("target", 2, 0),
        })

        _descriptor, full = _full_with_descriptor(descriptor=descriptor)
        scheduler = object.__new__(OnlineSchedulerBase)
        scheduler.joint_pipeline_ledger = PipelineLedger()
        scheduler.joint_work_ledger = WorkLedger()
        scheduler.work_ledger_absent_seen = False
        scheduler._apply_work_ledger({"work_ledger": full})

        wrong_local_parity = slot_row("source", 1, 0)
        for patch in (
                {"before": None, "after": wrong_local_parity},
                {"before": wrong_local_parity, "after": None}):
            bad_run = _pipeline_run(segment, 2, "consume_done", 1)
            bad_run["slot_patches"] = [patch]
            bad_delta = _empty_envelope(
                "delta", 1, base_watermark=1, base_version=1, tick=1)
            bad_delta.update({
                "logical_seq_start": 2,
                "logical_seq_end": 2,
                "logical_event_count": 1,
                "result_state_version": 2,
                "pipeline_transition_runs": [bad_run],
            })

            pipeline = scheduler.joint_pipeline_ledger
            work = scheduler.joint_work_ledger
            before_pipeline = pipeline.snapshot_state()
            before_work = work.snapshot_state()
            pipeline_marks = (
                pipeline.delivery_sequence, pipeline.watermark,
                pipeline.state_version)
            work_marks = (
                work.last_delivery_sequence, work.event_watermark,
                work.version)
            with self.assertRaisesRegex(
                    PipelineLedgerError,
                    "slot_index disagrees with two-slot tile parity"):
                scheduler._apply_work_ledger({"work_ledger": bad_delta})
            self.assertIs(pipeline.snapshot_state(), before_pipeline)
            self.assertIs(work.snapshot_state(), before_work)
            self.assertEqual(
                (pipeline.delivery_sequence, pipeline.watermark,
                 pipeline.state_version), pipeline_marks)
            self.assertEqual(
                (work.last_delivery_sequence, work.event_watermark,
                 work.version), work_marks)

    def test_target_data_ready_survives_until_consume_in_next_delivery(self):
        descriptor_wire = _descriptor_wire()
        # Use a nonzero global tile offset so selected slot parity must use
        # the descriptor's global ordinal rather than its local tile index.
        descriptor_wire["groups"][0]["tile_axis"][
            "global_tile_first"] = 4
        descriptor = PipelineDescriptor.from_wire(descriptor_wire)
        terminal_run = descriptor.consume_completion_runs[0]
        terminal_store_id = terminal_run.first_dependency_json_id + 1
        terminal_work_id = canonical_work_id(
            descriptor.identity.request_id, descriptor.identity.stage,
            descriptor.identity.generation, terminal_run.rank,
            terminal_store_id)
        terminal_payload = {
            "request_id": descriptor.identity.request_id,
            "stage": descriptor.identity.stage,
            "generation": descriptor.identity.generation,
            "rank": terminal_run.rank,
            "node_id": terminal_store_id,
            "operation": "compute",
            "total": {"ops": 1, "bytes": 0, "latency_ns": 1},
            "dependencies": [],
        }
        descriptor, full = _full_with_descriptor(node_events=(
            _node_event(2, "commit", terminal_work_id, terminal_payload),),
            descriptor=descriptor)
        scheduler = object.__new__(OnlineSchedulerBase)
        scheduler.joint_pipeline_ledger = PipelineLedger()
        scheduler.joint_work_ledger = WorkLedger()
        scheduler.work_ledger_absent_seen = False
        scheduler._apply_work_ledger({"work_ledger": full})
        segment = descriptor.identity.segment_id
        _assert_obligation_prefix_total(
            self, scheduler.joint_pipeline_ledger, segment)

        issue = _issue_delta(
            descriptor, delivery=1, watermark=2, issue_count=2)
        for transition in issue["pipeline_transition_runs"]:
            selector = transition["endpoint_selector"]
            if selector is None:
                continue
            global_tile = descriptor.tile(
                selector["group_ordinal"],
                selector["tile_ordinal"] - 1).global_tile_ordinal
            slot_index = (global_tile - 1) % 2
            for patch in transition["slot_patches"]:
                for row in (patch["before"], patch["after"]):
                    if row is not None:
                        row["slot_index"] = slot_index
        scheduler._apply_work_ledger({"work_ledger": issue})
        _assert_obligation_prefix_total(
            self, scheduler.joint_pipeline_ledger, segment)
        target_issue = issue["pipeline_transition_runs"][1]
        target_before = copy.deepcopy(
            target_issue["slot_patches"][0]["after"])
        active_obligation = copy.deepcopy(
            target_issue["obligation_patches"][0]["after"])
        target_run_index = target_issue["endpoint_run_index"] \
            if "endpoint_run_index" in target_issue else 1
        data_recv_run = descriptor.endpoint_runs[target_run_index]
        data_recv_store = descriptor.endpoint_id(
            target_run_index, data_recv_run.first_tile,
            data_recv_run.shard_ordinals[0], 0) + 1

        completed_cursor = copy.deepcopy(
            next(item["value"]
                 for item in issue["cursor_patches"][0]["runs"]
                 if item["run_index"] == target_run_index))
        completed_cursor.update({
            "completed_endpoint_count": 1,
            "cancelled_endpoint_count": 0,
            "remaining_endpoint_count": 2,
            "in_flight": False,
            "next_store_id": descriptor.endpoint_id(
                target_run_index, data_recv_run.first_tile + 1,
                data_recv_run.shard_ordinals[0], 0) + 1,
            "tile_offset": 1,
            "shard_index": 0,
            "leg_index": 0,
            "completed_out_of_order": [],
            "active_endpoints": [],
        })
        pending_obligation = copy.deepcopy(active_obligation)
        pending_obligation.update({
            "first_store_id": completed_cursor["next_store_id"],
            "remaining_endpoint_count": 2,
            "state": "pending",
        })
        selector = {
            "group_ordinal": data_recv_run.group_ordinal,
            "tile_ordinal": data_recv_run.first_tile,
            "shard_ordinal": data_recv_run.shard_ordinals[0],
            "endpoint": data_recv_run.legs[0].endpoint,
        }
        endpoint_amount = _expected_endpoint_amount(
            descriptor, target_run_index, 0)
        done = _pipeline_run(
            descriptor.identity.segment_id, 5, "endpoint_done", 2,
            cursor_start=2, cursor_end=3)
        done.update({
            "tile_first": selector["tile_ordinal"], "tile_stride": 0,
            "rank_first": data_recv_run.rank, "rank_stride": 0,
            "shard_first": selector["shard_ordinal"], "shard_stride": 0,
            "endpoint_selector": selector,
            "store_id_first": data_recv_store, "store_id_stride": 0,
            "legacy_context": {
                "request_id": data_recv_run.legacy_context.request_id,
                "stage": data_recv_run.legacy_context.stage,
                "generation": data_recv_run.legacy_context.generation,
            },
            "resource_leg": "endpoint",
            "resource_quantity_prefix": [
                {key: 0 if value is not None else None
                 for key, value in endpoint_amount.items()},
                endpoint_amount,
            ],
            "obligation_patches": [{
                "before": active_obligation,
                "after": pending_obligation,
            }],
        })
        data_ready = _pipeline_run(
            descriptor.identity.segment_id, 6, "data_ready", 2,
            cursor_start=3, cursor_end=3)
        data_ready.update({
            "tile_first": selector["tile_ordinal"], "tile_stride": 0,
            "rank_first": data_recv_run.rank, "rank_stride": 0,
            "shard_first": selector["shard_ordinal"], "shard_stride": 0,
            "endpoint_selector": selector,
            "store_id_first": data_recv_store, "store_id_stride": 0,
            "legacy_context": {
                "request_id": data_recv_run.legacy_context.request_id,
                "stage": data_recv_run.legacy_context.stage,
                "generation": data_recv_run.legacy_context.generation,
            },
        })
        data_ready_frame = _empty_envelope(
            "delta", 2, base_watermark=4, base_version=2, tick=2)
        data_ready_frame.update({
            "logical_seq_start": 5,
            "logical_seq_end": 6,
            "logical_event_count": 2,
            "result_state_version": 3,
            "pipeline_transition_runs": [done, data_ready],
            "cursor_patches": [{
                "segment_id": descriptor.identity.segment_id,
                "cursor_version": 3,
                "retired": False,
                "runs": [{"run_index": target_run_index,
                          "value": completed_cursor}],
            }],
        })
        original_selected = PipelineDescriptor.selected_shard_tile
        selected_calls = []

        def counted_selected(
                self, group_ordinal, local_tile_ordinal,
                group_shard_index):
            result = original_selected(
                self, group_ordinal, local_tile_ordinal,
                group_shard_index)
            selected_calls.append((
                group_ordinal, local_tile_ordinal, group_shard_index))
            return result

        with mock.patch.object(
                PipelineDescriptor, "selected_shard_tile",
                new=counted_selected):
            scheduler._apply_work_ledger({"work_ledger": data_ready_frame})
        expected_selected_calls = [
            (transition["endpoint_selector"]["group_ordinal"],
             transition["endpoint_selector"]["tile_ordinal"] - 1,
             transition["endpoint_selector"]["shard_ordinal"])
            for transition in data_ready_frame["pipeline_transition_runs"]
            if transition["endpoint_selector"] is not None]
        for cursor_patch in data_ready_frame["cursor_patches"]:
            for cursor_run in cursor_patch["runs"]:
                run = descriptor.endpoint_runs[cursor_run["run_index"]]
                value = cursor_run["value"]
                for active in value["active_endpoints"]:
                    if active["endpoint"] != "body_marker":
                        expected_selected_calls.append((
                            run.group_ordinal, active["tile_ordinal"] - 1,
                            active["shard_ordinal"]))
                if value["tile_offset"] < run.tile_count:
                    leg = run.legs[value["leg_index"]]
                    if leg.endpoint != "body_marker":
                        expected_selected_calls.append((
                            run.group_ordinal,
                            run.first_tile + value["tile_offset"] - 1,
                            run.shard_ordinals[value["shard_index"]]))
        self.assertEqual(selected_calls, expected_selected_calls)
        _assert_obligation_prefix_total(
            self, scheduler.joint_pipeline_ledger, segment)

        owner_tag = pipeline_ledger_module._target_owner_tag(
            target_before["owner"])
        ready_key = (
            descriptor.identity.buffer_domain_id, data_recv_run.rank,
            target_before["slot_index"], data_recv_run.shard_ordinals[0])
        self.assertEqual(
            scheduler.joint_pipeline_ledger._target_ready_shards_by_slot[
                ready_key], owner_tag)
        ready_global_tile = descriptor.tile(
            data_recv_run.group_ordinal,
            selector["tile_ordinal"] - 1).global_tile_ordinal
        self.assertEqual(ready_key[2], (ready_global_tile - 1) % 2)

        completion = _node_event(
            7, "completion", terminal_work_id, terminal_payload)
        release = _node_event(
            8, "release", terminal_work_id, terminal_payload)
        completion["tick"] = release["tick"] = 3
        terminal_frame = _empty_envelope(
            "delta", 3, base_watermark=6, base_version=3, tick=3)
        terminal_frame.update({
            "logical_seq_start": 7,
            "logical_seq_end": 8,
            "logical_event_count": 2,
            "result_state_version": 4,
            "node_transitions": [completion, release],
        })
        scheduler._apply_work_ledger({"work_ledger": terminal_frame})
        terminal_root = (
            scheduler.joint_pipeline_ledger
            ._completed_terminal_intervals_by_rank[terminal_run.rank])
        self.assertTrue(pipeline_ledger_module._completed_interval_contains(
            terminal_root, terminal_store_id))

        consumed = _pipeline_run(
            descriptor.identity.segment_id, 9, "consume_done", 4,
            cursor_start=3, cursor_end=3)
        target_after = copy.deepcopy(target_before)
        target_after.update({
            "occupied": False, "owner": None,
            "target_shards_to_consume": [],
            "consumed_target_shards": [],
        })
        consumed["slot_patches"] = [{
            "before": target_before, "after": target_after,
        }]
        consume_frame = _empty_envelope(
            "delta", 4, base_watermark=8, base_version=4, tick=4)
        consume_frame.update({
            "logical_seq_start": 9,
            "logical_seq_end": 9,
            "logical_event_count": 1,
            "result_state_version": 5,
            "pipeline_transition_runs": [consumed],
        })
        scheduler._apply_work_ledger({"work_ledger": consume_frame})
        self.assertNotIn(
            ready_key,
            scheduler.joint_pipeline_ledger._target_ready_shards_by_slot)

    def test_empty_delta_accepts_authoritative_resource_only_progress(self):
        ledger = PipelineLedger()
        ledger.apply_envelope(_empty_envelope(
            "full", 0, base_watermark=0, base_version=0))

        delta = _empty_envelope(
            "delta", 1, base_watermark=0, base_version=0, tick=1)
        delta["result_state_version"] = 1
        delta["active_resources"] = [_noc_resource(80, 1)]
        receipt = ledger.apply_envelope(delta)
        self.assertEqual(receipt.logical_event_count, 0)
        self.assertEqual(ledger.state_version, 1)
        self.assertEqual(ledger.snapshot_state()["active_resources"][0]["remaining"]["bytes"], 80)

        # A later authoritative service frame can change without a discrete
        # callback, but the visible-state version must still advance.
        bad = _empty_envelope(
            "delta", 2, base_watermark=0, base_version=1, tick=2)
        bad["result_state_version"] = 1
        bad["active_resources"] = [_noc_resource(60, 2)]
        before = ledger.snapshot_state()
        with self.assertRaisesRegex(PipelineLedgerError, "state change"):
            ledger.apply_envelope(bad)
        self.assertIs(ledger.snapshot_state(), before)
        self.assertEqual(ledger.watermark, 0)
        self.assertEqual(ledger.state_version, 1)
        self.assertEqual(ledger.snapshot_state()["active_resources"][0]["remaining"]["bytes"], 80)

    def test_unattributed_noc_keeps_legacy_empty_stage_flow_identity(self):
        ledger = PipelineLedger()
        full = _empty_envelope(
            "full", 0, base_watermark=0, base_version=0, tick=3)
        request_id, stage, generation, rank, flow_id = "noc", "", 0, 2, 42
        node_id = flow_id
        node_work_id = canonical_work_id(
            request_id, stage, generation, rank, node_id)
        full["active_resources"] = [{
            "resource_type": "noc_flow",
            "work_id": leg_work_id(node_work_id, f"noc:{flow_id}"),
            "request_id": request_id, "stage": stage,
            "generation": generation, "rank": rank, "node_id": node_id,
            "operation": "comm_send", "leg": f"noc:{flow_id}",
            "state": "Active",
            "total": {"ops": None, "bytes": 400, "latency_ns": None},
            "remaining": {"ops": None, "bytes": 250, "latency_ns": None},
            "progress_source": "authoritative", "service_unit": "bytes",
            "route": [7, 8], "as_of_tick": 3, "issued_tick": 1,
            "descriptor_segment_id": None, "endpoint_selector": None,
            "flow_id": flow_id, "current_rate_Bpns": 1.0,
            "last_rate_update_time": 2.0,
        }]
        full["result_state_version"] = 1
        ledger.apply_envelope(full)
        row = ledger.snapshot_state()["active_resources"][0]
        self.assertEqual((row["request_id"], row["stage"], row["generation"]),
                         ("noc", "", 0))
        self.assertEqual(row["work_id"], leg_work_id(node_work_id, "noc:42"))

    def test_cursor_rows_match_descriptor_emission_runs_and_store_ids(self):
        descriptor, full = _full_with_descriptor()
        segment = descriptor.identity.segment_id
        ledger = PipelineLedger()
        ledger.apply_envelope(full)
        self.assertEqual(
            ledger.snapshot_state()["authoritative_cursors"][0]["runs"][0]["next_store_id"],
            _initial_cursor_run(descriptor, segment, 0)["next_store_id"])

        malformed = copy.deepcopy(full)
        malformed["cursor_patches"][0]["runs"][0]["value"]["next_store_id"] += 1
        rejected = PipelineLedger()
        with self.assertRaisesRegex(PipelineLedgerError, "next_store_id"):
            rejected.apply_envelope(malformed)
        self.assertFalse(rejected.initialized)
        self.assertEqual(rejected.watermark, 0)

    def test_descriptor_cannot_finish_service_before_cursor_exhaustion(self):
        descriptor, full = _full_with_descriptor()
        ledger = PipelineLedger()
        ledger.apply_envelope(full)
        self.assertTrue(ledger.apply_envelope(copy.deepcopy(full)).replayed)

        segment = descriptor.identity.segment_id
        delta = _empty_envelope(
            "delta", 1, base_watermark=1, base_version=1, tick=1)
        delta.update({
            "logical_seq_end": 2,
            "logical_event_count": 1,
            "result_state_version": 2,
            "pipeline_transition_runs": [
                _pipeline_run(segment, 2, "service_done", 1)],
        })
        with self.assertRaisesRegex(PipelineLedgerError, "issue exhaustion"):
            ledger.apply_envelope(delta)
        self.assertIn(segment, ledger._descriptors)
        self.assertFalse(ledger._descriptor_state[segment]["service_done"])
        self.assertFalse(ledger._descriptor_state[segment]["retired"])

    def test_bad_node_suffix_keeps_both_ledgers_unchanged(self):
        work_id = "ordinary#decode:1:r0n9"
        commit_payload = {
            "request_id": "ordinary", "stage": "decode",
            "generation": 1, "rank": 0, "node_id": 9,
            "operation": "compute", "num_ops": 10, "bytes": 0,
            "runtime_ns": 1, "tensor_size": 0, "parents": [],
        }
        _descriptor, envelope = _full_with_descriptor(node_events=(
            _node_event(2, "commit", work_id, commit_payload),
            _node_event(3, "release", "unknown#decode:1:r0n1"),
        ))
        # The last event is structurally valid but semantically references an
        # unknown node.  The valid commit prefix remains only in the detached
        # COW shadow; neither accepted ledger may publish it.
        scheduler = object.__new__(OnlineSchedulerBase)
        scheduler.joint_pipeline_ledger = PipelineLedger()
        scheduler.joint_work_ledger = WorkLedger()
        scheduler.work_ledger_absent_seen = True
        with self.assertRaises(WorkLedgerError):
            scheduler._apply_work_ledger({"work_ledger": envelope})
        self.assertFalse(scheduler.joint_pipeline_ledger.initialized)
        self.assertEqual(scheduler.joint_pipeline_ledger.watermark, 0)
        self.assertEqual(scheduler.joint_work_ledger._works, {})
        self.assertEqual(scheduler.joint_work_ledger.version, 0)
        self.assertEqual(scheduler.joint_work_ledger.full_load_count, 0)
        self.assertTrue(scheduler.work_ledger_absent_seen)

    def test_retirement_checks_dependency_pin_at_sequence_prefix_atomically(self):
        descriptor = PipelineDescriptor.from_wire(_descriptor_wire())
        endpoint_store_id = descriptor.endpoint_runs[0].first_json_id + 1
        dependent_id = "dependent#decode:1:r0n90"
        dependency_id = canonical_work_id(
            "req", "decode", 1, 0, endpoint_store_id)
        child_event = _node_event(2, "commit", dependent_id, {
            "request_id": "dependent", "stage": "decode",
            "generation": 1, "rank": 0, "node_id": 90,
            "operation": "compute",
            "total": {"ops": 10, "bytes": 0, "latency_ns": 1},
            "dependencies": [dependency_id],
        })
        descriptor, full = _full_with_descriptor(node_events=(child_event,))
        # The publisher stages ordinary graph commits before the declaration's
        # descriptor_commit transition. The immutable declaration is already
        # present in this envelope, so its exact endpoint can be pinned while
        # the descriptor is still inactive.
        full["node_transitions"][0]["logical_seq"] = 1
        full["pipeline_transition_runs"][0]["first_logical_seq"] = 2
        scheduler = object.__new__(OnlineSchedulerBase)
        scheduler.joint_pipeline_ledger = PipelineLedger()
        scheduler.joint_work_ledger = WorkLedger()
        scheduler.work_ledger_absent_seen = False
        scheduler._apply_work_ledger({"work_ledger": full})
        _assert_obligation_prefix_total(
            self, scheduler.joint_pipeline_ledger,
            descriptor.identity.segment_id)

        # request_close cancels the unissued tails, while preserving the
        # ordinary dependency pin that must still block descriptor retirement.
        segment = descriptor.identity.segment_id
        close_rows = []
        for item in full["cursor_patches"][0]["runs"]:
            row = copy.deepcopy(item["value"])
            run = descriptor.endpoint_runs[item["run_index"]]
            total = run.tile_count * len(run.shard_ordinals) * len(run.legs)
            completed = row["completed_endpoint_count"] + sum(
                interval["end_ordinal_exclusive"]
                - interval["first_ordinal"]
                for interval in row["completed_out_of_order"])
            row.update({
                "cancelled_endpoint_count": total - completed,
                "remaining_endpoint_count": 0,
                "in_flight": False,
                "next_store_id": None,
                "tile_offset": run.tile_count,
                "shard_index": 0,
                "leg_index": 0,
                "active_endpoints": [],
            })
            close_rows.append({"run_index": item["run_index"], "value": row})
        initial_obligations = [
            patch["after"]
            for patch in full["pipeline_transition_runs"][0][
                "obligation_patches"]
            if patch["after"] is not None
        ]
        close = _empty_envelope(
            "delta", 1, base_watermark=2, base_version=1, tick=1)
        close.update({
            "logical_seq_start": 3,
            "logical_seq_end": 3,
            "logical_event_count": 1,
            "result_state_version": 2,
            "pipeline_transition_runs": [
                _pipeline_run(segment, 3, "request_close", 1,
                              cursor_start=0, cursor_end=1)],
            "cursor_patches": [{
                "segment_id": segment,
                "cursor_version": 1,
                "retired": False,
                "runs": close_rows,
            }],
        })
        close["pipeline_transition_runs"][0]["obligation_patches"] = [
            {"before": row, "after": None}
            for row in sorted(initial_obligations,
                              key=lambda item: item["obligation_id"])
        ]
        scheduler._apply_work_ledger({"work_ledger": close})
        _assert_obligation_prefix_total(
            self, scheduler.joint_pipeline_ledger, segment)
        self.assertEqual(
            scheduler.joint_pipeline_ledger
            ._obligation_remaining_by_segment.get(segment, 0), 0)
        before_pipeline = scheduler.joint_pipeline_ledger.snapshot_state()
        before_work_version = scheduler.joint_work_ledger.version

        retire = _empty_envelope(
            "delta", 2, base_watermark=3, base_version=2, tick=1)
        retire.update({
            "logical_seq_start": 4,
            "logical_seq_end": 6,
            "logical_event_count": 3,
            "result_state_version": 3,
            # This ordinary pin release is deliberately after retirement in
            # the merged stream; it cannot legalize an earlier retire prefix.
            "node_transitions": [
                {**_node_event(6, "release", dependent_id), "tick": 1}],
            "pipeline_transition_runs": [
                _pipeline_run(segment, 4, "settle_done", 1,
                              cursor_start=1, cursor_end=2),
                _pipeline_run(segment, 5, "descriptor_retire", 1,
                              cursor_start=2, cursor_end=3),
            ],
            "cursor_patches": [{
                "segment_id": segment,
                "cursor_version": 3,
                "retired": True,
                "runs": [{"run_index": index, "value": None}
                         for index in range(len(descriptor.endpoint_runs))],
            }],
        })
        with self.assertRaisesRegex(PipelineLedgerError, "dependency pin"):
            scheduler._apply_work_ledger({"work_ledger": retire})
        self.assertIs(scheduler.joint_pipeline_ledger.snapshot_state(),
                      before_pipeline)
        self.assertEqual(scheduler.joint_pipeline_ledger.delivery_sequence, 1)
        self.assertEqual(scheduler.joint_pipeline_ledger.state_version, 2)
        self.assertEqual(scheduler.joint_work_ledger.version, before_work_version)
        self.assertIn(dependent_id, scheduler.joint_work_ledger._works)

    def test_completion_removes_dependency_pin_from_mutated_cow_work(self):
        descriptor = PipelineDescriptor.from_wire(_descriptor_wire())
        parent_store_id = descriptor.endpoint_runs[0].first_json_id + 1
        parent_id = canonical_work_id(
            "req", "decode", 1, 0, parent_store_id)
        parent_identity = {
            "request_id": "req", "stage": "decode",
            "generation": 1, "rank": 0,
            "node_id": parent_store_id, "operation": "comm_send",
        }
        dependent_id = "dependent#decode:1:r0n90"
        dependent_identity = {
            "request_id": "dependent", "stage": "decode",
            "generation": 1, "rank": 0, "node_id": 90,
            "operation": "compute",
        }
        descriptor, full = _full_with_descriptor(node_events=(
            _node_event(2, "commit", parent_id, {
                **parent_identity,
                "total": {"ops": 0, "bytes": 1, "latency_ns": 0},
            }),
            _node_event(3, "commit", dependent_id, {
                **dependent_identity,
                "total": {"ops": 10, "bytes": 0, "latency_ns": 1},
                "dependencies": [parent_id],
            }),
        ))
        scheduler = object.__new__(OnlineSchedulerBase)
        scheduler.joint_pipeline_ledger = PipelineLedger()
        scheduler.joint_work_ledger = WorkLedger()
        scheduler.work_ledger_absent_seen = False
        scheduler._apply_work_ledger({"work_ledger": full})

        segment = descriptor.identity.segment_id
        self.assertEqual(
            scheduler.joint_pipeline_ledger
            ._unfinished_dependency_pins_by_segment.get(segment), 1)

        # The exact pipeline parent completes and then the still-active child
        # issues. Completion removes the pin at that stream prefix, so the
        # child no longer keeps the retired parent descriptor alive.
        parent_completion = _empty_envelope(
            "delta", 1, base_watermark=3, base_version=1, tick=1)
        parent_completion.update({
            "logical_seq_start": 4,
            "logical_seq_end": 6,
            "logical_event_count": 3,
            "result_state_version": 2,
            "node_transitions": [
                _node_event(4, "issue", parent_id, parent_identity),
                _node_event(5, "completion", parent_id, parent_identity),
                _node_event(6, "issue", dependent_id, dependent_identity)],
        })
        scheduler._apply_work_ledger({"work_ledger": parent_completion})
        self.assertEqual(
            scheduler.joint_pipeline_ledger
            ._unfinished_dependency_pins_by_segment.get(segment, 0), 0)
        self.assertEqual(
            scheduler.joint_pipeline_ledger
            ._unfinished_dependency_waiters_by_endpoint.get(
                (0, parent_store_id, segment), 0), 0)
        self.assertEqual(
            scheduler.joint_work_ledger._works[dependent_id].state,
            "issued")

        # Completing the child after its parent was already proven terminal
        # must not decrement the pin a second time.
        completion = _empty_envelope(
            "delta", 2, base_watermark=6, base_version=2, tick=1)
        completion.update({
            "logical_seq_start": 7,
            "logical_seq_end": 7,
            "logical_event_count": 1,
            "result_state_version": 3,
            "node_transitions": [
                {**_node_event(7, "completion", dependent_id,
                               dependent_identity), "tick": 1}],
        })
        scheduler._apply_work_ledger({"work_ledger": completion})
        self.assertEqual(
            scheduler.joint_pipeline_ledger
            ._unfinished_dependency_pins_by_segment.get(segment, 0), 0)
        self.assertEqual(
            scheduler.joint_work_ledger._works[dependent_id].state,
            "completed")

    def test_bad_suffix_does_not_publish_dependency_terminal_resolution(self):
        descriptor = PipelineDescriptor.from_wire(_descriptor_wire())
        parent_store_id = descriptor.endpoint_runs[0].first_json_id + 1
        parent_id = canonical_work_id(
            "req", "decode", 1, 0, parent_store_id)
        parent_identity = {
            "request_id": "req", "stage": "decode",
            "generation": 1, "rank": 0,
            "node_id": parent_store_id, "operation": "comm_send",
        }
        dependent_id = "dependent#decode:1:r0n92"
        descriptor, full = _full_with_descriptor(node_events=(
            _node_event(2, "commit", parent_id, {
                **parent_identity,
                "total": {"ops": 0, "bytes": 1, "latency_ns": 0},
            }),
            _node_event(3, "commit", dependent_id, {
                "request_id": "dependent", "stage": "decode",
                "generation": 1, "rank": 0, "node_id": 92,
                "operation": "compute",
                "total": {"ops": 1, "bytes": 0, "latency_ns": 1},
                "dependencies": [parent_id],
            }),
        ))
        scheduler = object.__new__(OnlineSchedulerBase)
        scheduler.joint_pipeline_ledger = PipelineLedger()
        scheduler.joint_work_ledger = WorkLedger()
        scheduler.work_ledger_absent_seen = False
        scheduler._apply_work_ledger({"work_ledger": full})
        ledger = scheduler.joint_pipeline_ledger
        segment = descriptor.identity.segment_id
        self.assertEqual(
            ledger._unfinished_dependency_pins_by_segment.get(segment), 1)
        before_pipeline = ledger.snapshot_state()
        before_work_version = scheduler.joint_work_ledger.version

        bad_suffix = _empty_envelope(
            "delta", 1, base_watermark=3, base_version=1, tick=1)
        bad_suffix.update({
            "logical_seq_start": 4,
            "logical_seq_end": 5,
            "logical_event_count": 2,
            "result_state_version": 2,
            "node_transitions": [
                _node_event(4, "completion", parent_id, parent_identity),
                _node_event(5, "release", "unknown#decode:1:r0n99")],
        })
        with self.assertRaises(WorkLedgerError):
            scheduler._apply_work_ledger({"work_ledger": bad_suffix})
        self.assertIs(ledger.snapshot_state(), before_pipeline)
        self.assertEqual(ledger.delivery_sequence, 0)
        self.assertEqual(
            ledger._unfinished_dependency_pins_by_segment.get(segment), 1)
        self.assertEqual(
            ledger._unfinished_dependency_waiters_by_endpoint.get(
                (0, parent_store_id, segment)), 1)
        self.assertNotIn(
            0, ledger._completed_terminal_intervals_by_rank)
        self.assertEqual(scheduler.joint_work_ledger.version,
                         before_work_version)
        self.assertEqual(
            scheduler.joint_work_ledger._works[parent_id].state,
            "committed")

    def test_retired_dependency_requires_exact_terminal_provenance(self):
        descriptor = PipelineDescriptor.from_wire(_descriptor_wire())
        segment = descriptor.identity.segment_id
        run = descriptor.endpoint_runs[0]
        parent_store_id = run.first_json_id + 1
        parent_id = canonical_work_id(
            run.legacy_context.request_id, run.legacy_context.stage,
            run.legacy_context.generation, run.rank, parent_store_id)
        dependent_id = "dependent#decode:1:r0n91"
        dependent_payload = {
            "request_id": "dependent", "stage": "decode",
            "generation": 1, "rank": 0, "node_id": 91,
            "operation": "compute",
            "total": {"ops": 1, "bytes": 0, "latency_ns": 1},
            "dependencies": [parent_id],
        }

        # Isolate only the future-reference rule after retirement. Actual
        # cursor retirement is covered by the transition-prefix tests.
        # An absent terminal interval models canceled or unissued work: neither
        # path supplies physical completion evidence for the future reference.
        for terminal_rank, terminal_id, accepted in (
                (None, None, False),
                (run.rank + 1, parent_store_id, False),
                (run.rank, parent_store_id - 1, False),
                (run.rank, parent_store_id, True)):
            with self.subTest(terminal_rank=terminal_rank,
                              terminal_id=terminal_id):
                scheduler = object.__new__(OnlineSchedulerBase)
                scheduler.joint_pipeline_ledger = PipelineLedger()
                scheduler.joint_work_ledger = WorkLedger()
                scheduler.work_ledger_absent_seen = False
                scheduler._apply_work_ledger({"work_ledger": _full_with_descriptor()[1]})
                ledger = scheduler.joint_pipeline_ledger
                retired_state = dict(ledger._descriptor_state[segment])
                retired_state["retired"] = True
                ledger._descriptor_state[segment] = retired_state
                ledger._active_segments.discard(segment)
                if terminal_rank is not None:
                    ledger._completed_terminal_intervals_by_rank[
                        terminal_rank] = (
                            pipeline_ledger_module._completed_interval_add(
                                ledger._completed_terminal_intervals_by_rank.get(
                                    terminal_rank), terminal_id))

                commit = _empty_envelope(
                    "delta", 1, base_watermark=1,
                    base_version=1, tick=1)
                commit.update({
                    "logical_seq_start": 2,
                    "logical_seq_end": 2,
                    "logical_event_count": 1,
                    "result_state_version": 2,
                    "node_transitions": [
                        _node_event(2, "commit", dependent_id,
                                   dependent_payload)],
                })
                if accepted:
                    scheduler._apply_work_ledger({"work_ledger": commit})
                    self.assertIn(
                        dependent_id, scheduler.joint_work_ledger._works)
                    self.assertEqual(
                        ledger._unfinished_dependency_pins_by_segment.get(
                            segment, 0), 0)
                else:
                    with self.assertRaisesRegex(
                            PipelineLedgerError,
                            "incomplete endpoint in a retired"):
                        scheduler._apply_work_ledger(
                            {"work_ledger": commit})
                    self.assertNotIn(
                        dependent_id, scheduler.joint_work_ledger._works)

    def test_dependency_index_matches_physical_and_virtual_endpoint_ids(self):
        descriptor = PipelineDescriptor.from_wire(_descriptor_wire())
        _descriptor, full = _full_with_descriptor()
        ledger = PipelineLedger()
        ledger.apply_envelope(full)
        index = ledger._dependency_endpoint_index
        segment = descriptor.identity.segment_id

        # Rank 0's first endpoint is the physical anchor (NodeStore id 52);
        # subsequent affine ids 54 and 56 are reserved virtual endpoints.
        self.assertEqual(
            pipeline_ledger_module._dependency_segment(
                "req#decode:1:r0n52", index), segment)
        self.assertEqual(
            pipeline_ledger_module._dependency_segment(
                "req#decode:1:r0n54", index), segment)

        # A dependency must retain the exact request, descriptor identity,
        # endpoint-owner rank, and affine leg stream used by the index.
        for dependency in (
                "other#decode:1:r0n52",  # wrong request
                "req#decode:2:r0n52",    # wrong generation
                "req#decode:1:r1n52",    # wrong owner rank
                "req#decode:1:r0n55",    # source rank with receiver-leg id
                "req#decode:1:r0n57",    # outside rank-0 endpoint progression
        ):
            with self.subTest(dependency=dependency):
                self.assertIsNone(
                    pipeline_ledger_module._dependency_segment(
                        dependency, index))


class SparsePrepareScalingTest(unittest.TestCase):
    """Count actual touched-row reads while unrelated descriptor history grows."""

    def test_fixed_transition_count_does_not_scan_unrelated_descriptors(self):
        measured = []
        for descriptor_count in (8, 32, 128):
            descriptor, _full = _full_with_descriptors(descriptor_count)
            frame = _issue_delta(
                descriptor, delivery=1, watermark=descriptor_count,
                issue_count=1)
            measured.append(_measure_sparse_prepare(frame, descriptor_count))
        row_accesses = [item["row_accesses"] for item in measured]
        encoded_bytes = [item["encoded_bytes"] for item in measured]
        normalized_bytes = [
            item["header_normalized_payload_bytes"] for item in measured]
        self.assertEqual(row_accesses, [row_accesses[0]] * len(row_accesses))
        self.assertEqual(
            normalized_bytes,
            [normalized_bytes[0]] * len(normalized_bytes),
            "payload sizes should stay fixed after only declared sequence "
            "metadata is normalized; raw_bytes={} normalized_bytes={}".format(
                encoded_bytes, normalized_bytes))
        self.assertGreater(row_accesses[0], 0)

    def test_changed_endpoint_rows_scale_with_transition_count(self):
        descriptor_count = 32
        descriptor, _full = _full_with_descriptors(descriptor_count)
        measured = []
        for issue_count in (1, 2, 4):
            frame = _issue_delta(
                descriptor, delivery=1, watermark=descriptor_count,
                issue_count=issue_count)
            measured.append(_measure_sparse_prepare(frame, descriptor_count))
        row_accesses = [item["row_accesses"] for item in measured]
        encoded_bytes = [item["encoded_bytes"] for item in measured]
        self.assertLess(row_accesses[0], row_accesses[1])
        self.assertLess(row_accesses[1], row_accesses[2])
        self.assertLess(encoded_bytes[0], encoded_bytes[1])
        self.assertLess(encoded_bytes[1], encoded_bytes[2])

    def test_replayed_numeric_type_change_is_rejected(self):
        ledger = PipelineLedger()
        full = _empty_envelope(
            "full", 0, base_watermark=0, base_version=0)
        ledger.apply_envelope(full)
        changed = copy.deepcopy(full)
        changed["tick"] = 0.0
        with self.assertRaises(PipelineLedgerError):
            ledger.apply_envelope(changed)
        self.assertEqual(ledger.delivery_sequence, 0)
        self.assertEqual(ledger.state_version, 0)


class CompletedCursorIntervalCostTest(unittest.TestCase):
    """Exact interval semantics plus observed deterministic treap work counts.

    The SplitMix treap gives expected/observed path costs for these fixtures;
    these measurements do not claim adversarial worst-case logarithmic depth.
    """

    @staticmethod
    def _old_list_contains(ranges, ordinal):
        visits = 0
        for first, end in ranges:
            visits += 1
            if first <= ordinal < end:
                return True, visits
        return False, visits

    def test_sorted_range_membership_reads_only_binary_search_candidates(self):
        class ReadCountingSequence:
            def __init__(self, rows):
                self.rows = rows
                self.row_reads = 0

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                self.row_reads += 1
                return self.rows[index]

        boundary_ranges = ((0, 3), (8, 11), (15, 20))
        for ordinal in (0, 2, 3, 7, 8, 10, 11, 19, 20):
            actual = pipeline_ledger_module._completed_range_contains_half_open_sorted(
                boundary_ranges, ordinal)
            expected, _visits = self._old_list_contains(
                boundary_ranges, ordinal)
            self.assertEqual(actual, expected)

        fixed_active_endpoints = 8
        for history in (8, 32, 128):
            ranges = tuple((10 + index * 4, 11 + index * 4)
                           for index in range(history))
            binary_row_reads = 0
            old_list_row_reads = 0
            for query_index in range(fixed_active_endpoints):
                ordinal = ranges[-1][1] + 2 + query_index
                counted = ReadCountingSequence(ranges)
                actual = (
                    pipeline_ledger_module
                    ._completed_range_contains_half_open_sorted(
                        counted, ordinal))
                expected, old_reads = self._old_list_contains(
                    ranges, ordinal)
                self.assertEqual(actual, expected)
                self.assertEqual(actual, False)
                self.assertLessEqual(
                    counted.row_reads, history.bit_length() + 1)
                binary_row_reads += counted.row_reads
                old_list_row_reads += old_reads
            self.assertLess(binary_row_reads, old_list_row_reads)
            print(
                "[completed-cursor-parser-range-membership-scale] "
                f"A={fixed_active_endpoints} H={history} "
                f"bisect_row_reads={binary_row_reads} "
                f"old_list_predicate_row_reads={old_list_row_reads}")

    @staticmethod
    def _old_list_add(ranges, completed, total, first, end, work=None):
        if first < completed or end <= first or end > total:
            raise ValueError("duplicate or invalid completed range")
        for start, range_end in ranges:
            if work is not None:
                work["old_overlap_predicate_rows"] = (
                    work.get("old_overlap_predicate_rows", 0) + 1)
            if start < end and first < range_end:
                raise ValueError("duplicate or invalid completed range")
        merged = []
        for start, range_end in sorted(ranges + [(first, end)]):
            if work is not None:
                work["old_merge_scan_rows"] = (
                    work.get("old_merge_scan_rows", 0) + 1)
            if merged and start == merged[-1][1]:
                merged[-1] = (merged[-1][0], range_end)
            else:
                merged.append((start, range_end))
        while merged and merged[0][0] == completed:
            _start, completed = merged.pop(0)
        return merged, completed

    def test_seed_membership_and_export_match_list_oracle_at_scales(self):
        measured = []
        fixed_queries = 8
        for history in (8, 32, 128):
            ranges = [(10 + index * 4, 11 + index * 4)
                      for index in range(history)]
            seed_work = {}
            root, out_of_order_count = (
                pipeline_ledger_module._completed_interval_build_from_half_open(
                    ranges, seed_work))
            old_count = sum(end - first for first, end in ranges)
            self.assertEqual(out_of_order_count, old_count)
            self.assertEqual(seed_work["seed_interval_rows"], history)
            self.assertEqual(seed_work["seed_build_steps"], history)
            self.assertEqual(seed_work["seed_tree_nodes"], history)
            self.assertLessEqual(seed_work["seed_stack_pop_steps"], history)
            self.assertEqual(seed_work["seed_postorder_visits"], history * 2)

            export_work = {}
            self.assertEqual(
                pipeline_ledger_module._completed_interval_export_half_open(
                    root, export_work), tuple(ranges))
            self.assertEqual(export_work["export_node_visits"], history)

            # Probe a missing point after the final range so the independent
            # old-list oracle visits all H rows on each fixed-T query.
            query = ranges[-1][1] + 2
            new_work = {}
            old_visits = 0
            for _ in range(fixed_queries):
                actual = pipeline_ledger_module._completed_interval_contains(
                    root, query, new_work)
                expected, row_visits = self._old_list_contains(ranges, query)
                old_visits += row_visits
                self.assertEqual(actual, expected)
            new_visits = new_work["tree_query_node_visits"]
            self.assertLess(new_visits, old_visits)
            measured.append((history, new_visits, old_visits))
            print(
                "[completed-cursor-membership-scale] "
                f"T={fixed_queries} H={history} tree_node_visits={new_visits} "
                f"old_list_row_visits={old_visits}")

        self.assertEqual([row[0] for row in measured], [8, 32, 128])

    def test_fixed_history_range_updates_and_membership_scale_with_t(self):
        history = 128
        base_start = 10000
        base_ranges = [(base_start + index * 100,
                        base_start + index * 100 + 1)
                       for index in range(history)]
        measured = []
        for transition_count in (8, 32, 128):
            work = {}
            root, _ooo_count = (
                pipeline_ledger_module._completed_interval_build_from_half_open(
                    base_ranges, work))
            completed_prefix = 0
            completed_total = history
            ranges = list(base_ranges)
            old_list_work = {}
            for index in range(transition_count):
                end = base_start - index
                first = end - 1
                previous_completed = completed_prefix
                ranges, expected_completed = self._old_list_add(
                    ranges, previous_completed,
                    base_ranges[-1][1] + transition_count + 1,
                    first, end, old_list_work)
                root, actual_completed = (
                    pipeline_ledger_module._completed_interval_add_range(
                        root, previous_completed,
                        base_ranges[-1][1] + transition_count + 1,
                        first, end, work))
                self.assertEqual(actual_completed, expected_completed)
                completed_prefix = expected_completed
                completed_total += end - first
                self.assertEqual(
                    pipeline_ledger_module._completed_interval_export_half_open(
                        root), tuple(ranges))
                self.assertEqual(completed_total,
                                 completed_prefix + sum(
                                     last - begin for begin, last in ranges))
                self.assertEqual(len(ranges), history)

            tree_visits = (work.get("tree_query_node_visits", 0)
                           + work.get("tree_update_node_visits", 0))
            old_overlap_rows = old_list_work["old_overlap_predicate_rows"]
            old_merge_rows = old_list_work["old_merge_scan_rows"]
            old_list_rows_visited = old_overlap_rows + old_merge_rows
            self.assertEqual(work["range_updates"], transition_count)
            self.assertEqual(old_overlap_rows, transition_count * history)
            self.assertEqual(old_merge_rows, transition_count * (history + 1))
            self.assertGreater(tree_visits, 0)
            self.assertLess(tree_visits, old_list_rows_visited)
            measured.append((transition_count, tree_visits,
                             old_list_rows_visited, old_overlap_rows,
                             old_merge_rows))
            print(
                "[completed-cursor-range-update-scale] "
                f"H={history} T={transition_count} "
                f"tree_node_visits={tree_visits} "
                f"old_overlap_predicate_rows={old_overlap_rows} "
                f"old_merge_scan_rows={old_merge_rows} "
                f"old_list_row_visits={old_list_rows_visited}")
        self.assertLess(measured[0][1], measured[1][1])
        self.assertLess(measured[1][1], measured[2][1])

    def test_range_union_prefix_duplicate_overlap_and_u64_boundaries(self):
        adjacent_wire_ranges = [(1, 2), (2, 3), (7, 8)]
        adjacent_root, adjacent_count = (
            pipeline_ledger_module._completed_interval_build_from_half_open(
                adjacent_wire_ranges))
        self.assertEqual(adjacent_count, 3)
        self.assertEqual(adjacent_wire_ranges, [(1, 2), (2, 3), (7, 8)])
        self.assertEqual(
            pipeline_ledger_module._completed_interval_export_half_open(
                adjacent_root), ((1, 3), (7, 8)))

        initial = [(2, 3), (6, 7), (9, 11)]
        root, out_of_order_count = (
            pipeline_ledger_module._completed_interval_build_from_half_open(
                initial))
        completed = 0
        completed_total = out_of_order_count
        total = 20
        self.assertEqual(out_of_order_count, 4)

        for first, end in ((3, 6), (0, 2), (7, 9)):
            old_root = root
            old_ranges = tuple(initial)
            previous_completed = completed
            initial, expected_completed = self._old_list_add(
                initial, previous_completed, total, first, end)
            root, actual_completed = (
                pipeline_ledger_module._completed_interval_add_range(
                    root, previous_completed, total, first, end))
            self.assertEqual(actual_completed, expected_completed)
            completed = expected_completed
            completed_total += end - first
            self.assertEqual(
                pipeline_ledger_module._completed_interval_export_half_open(
                    root), tuple(initial))
            self.assertEqual(
                pipeline_ledger_module._completed_interval_export_half_open(
                    old_root), old_ranges)
            self.assertEqual(
                completed_total,
                completed + sum(last - begin for begin, last in initial))
            for ordinal in range(total):
                expected = (ordinal < completed or any(
                    begin <= ordinal < end for begin, end in initial))
                actual = (ordinal < completed or
                          pipeline_ledger_module._completed_interval_contains(
                              root, ordinal))
                self.assertEqual(actual, expected)

        self.assertEqual(completed, 11)
        self.assertEqual(initial, [])

        overlap_ranges = [(4, 6), (10, 11)]
        overlap_root, _ = (
            pipeline_ledger_module._completed_interval_build_from_half_open(
                overlap_ranges))
        before_root = overlap_root
        with self.assertRaisesRegex(PipelineLedgerError, "overlaps"):
            pipeline_ledger_module._completed_interval_add_range(
                overlap_root, 0, 20, 5, 7)
        self.assertIs(overlap_root, before_root)
        with self.assertRaisesRegex(PipelineLedgerError, "duplicate"):
            pipeline_ledger_module._completed_interval_add_range(
                overlap_root, 6, 20, 5, 7)
        self.assertIs(overlap_root, before_root)
        with self.assertRaisesRegex(PipelineLedgerError, "outside"):
            pipeline_ledger_module._completed_interval_add_range(
                overlap_root, 0, 20, 19, 21)
        self.assertIs(overlap_root, before_root)

        u64_max = pipeline_ledger_module.U64_MAX
        boundary_root, prefix = pipeline_ledger_module._completed_interval_add_range(
            None, u64_max - 2, u64_max, u64_max - 2, u64_max)
        self.assertIsNone(boundary_root)
        self.assertEqual(prefix, u64_max)
        bounded_root, bounded_count = (
            pipeline_ledger_module._completed_interval_build_from_half_open(
                [(u64_max - 2, u64_max)]))
        self.assertEqual(bounded_count, 2)
        self.assertEqual(
            pipeline_ledger_module._completed_interval_export_half_open(
                bounded_root), ((u64_max - 2, u64_max),))
        with self.assertRaisesRegex(PipelineLedgerError, "outside"):
            pipeline_ledger_module._completed_interval_add_range(
                bounded_root, 0, u64_max, u64_max, u64_max + 1)
        self.assertEqual(
            pipeline_ledger_module._completed_interval_export_half_open(
                bounded_root), ((u64_max - 2, u64_max),))


class RuntimeMappingCompatibilityTest(unittest.TestCase):
    def test_mapping_forms_keep_pipeline_parse_freeze_and_typed_compare(self):
        pipeline_ledger_module._RuntimeMapping.register(_TestVirtualMapping)
        factories = (
            ("dict", dict),
            ("dict subclass", _TestDictSubclass),
            ("UserDict", UserDict),
            ("MappingProxyType", MappingProxyType),
            ("custom Mapping", _TestCustomMapping),
            ("virtual Mapping", _TestVirtualMapping),
        )
        values = {"value": 1, "nested": {"floating": 1.0}}
        expected_frozen = pipeline_ledger_module._freeze_dto(values)
        expected_resources = pipeline_ledger_module._parse_resources(
            [_noc_resource(80, 1)], {}, "active_resources")
        for name, factory in factories:
            with self.subTest(mapping_type=name):
                wrapped = _test_mapping_tree(values, factory)
                parsed = pipeline_ledger_module._mapping(
                    wrapped, "probe", frozenset({"value", "nested"}))
                self.assertIs(parsed, wrapped)
                self.assertTrue(
                    pipeline_ledger_module._typed_equal(values, wrapped))
                self.assertEqual(
                    pipeline_ledger_module._freeze_dto(wrapped),
                    expected_frozen)

                resource_row = _test_mapping_tree(
                    _noc_resource(80, 1), factory)
                # Virtual ABC registration supplies no Mapping equality
                # mixin. Compare parsed protocol values with the same strict
                # structural rule instead of assuming wrapper __eq__.
                self.assertTrue(pipeline_ledger_module._typed_equal(
                    pipeline_ledger_module._parse_resources(
                        [resource_row], {}, "active_resources"),
                    expected_resources))

        self.assertFalse(pipeline_ledger_module._typed_equal(
            {"value": 1}, {"value": 1.0}))
        self.assertFalse(pipeline_ledger_module._typed_equal(
            {"value": True}, {"value": 1}))
        self.assertFalse(pipeline_ledger_module._typed_equal(
            {"value": float("nan")}, {"value": float("nan")}))
        proxy = MappingProxyType({
            "value": 1, "nested": MappingProxyType({"floating": 1.0})})
        self.assertTrue(pipeline_ledger_module._typed_equal(values, proxy))
        self.assertTrue(pipeline_ledger_module._typed_equal(proxy, values))
        self.assertTrue(pipeline_ledger_module._typed_equal(
            [1, {"nested": True}], (1, MappingProxyType({"nested": True}))))
        with self.assertRaisesRegex(PipelineLedgerError, "must be an object"):
            pipeline_ledger_module._mapping(
                [], "probe", frozenset({"value"}))
        with self.assertRaisesRegex(
                PipelineLedgerError, "pipeline ledger envelope must be an object"):
            PipelineLedger().validate_envelope([])

    def test_class_spoof_matches_legacy_typing_mapping_consumer(self):
        spoofed_values = (
            mock.Mock(spec=_TestMapping),
            _TestSpoofedMappingClass(),
        )
        for value in spoofed_values:
            with self.subTest(spoof_type=type(value).__name__):
                # Bare typing.Mapping's legacy instance check uses the real
                # type(obj), while ABC isinstance may honor this spoofed class.
                self.assertTrue(isinstance(value, _TestMapping))
                self.assertFalse(isinstance(value, _LegacyTypingMapping))
                self.assertFalse(issubclass(
                    type(value), pipeline_ledger_module._RuntimeMapping))
                self.assertFalse(pipeline_ledger_module._typed_equal(
                    value, {}))
                self.assertIs(pipeline_ledger_module._freeze_dto(value), value)
                with self.assertRaisesRegex(
                        PipelineLedgerError, "must be an object"):
                    pipeline_ledger_module._mapping(
                        value, "probe", frozenset())

    def test_bad_numeric_mapping_preserves_both_ledger_states(self):
        scheduler = object.__new__(OnlineSchedulerBase)
        scheduler.joint_pipeline_ledger = PipelineLedger()
        scheduler.joint_work_ledger = WorkLedger()
        scheduler.work_ledger_absent_seen = False
        scheduler._apply_work_ledger({"work_ledger": _empty_envelope(
            "full", 0, base_watermark=0, base_version=0)})

        pipeline_before = scheduler.joint_pipeline_ledger.snapshot_state()
        pipeline_versions_before = (
            scheduler.joint_pipeline_ledger.delivery_sequence,
            scheduler.joint_pipeline_ledger.watermark,
            scheduler.joint_pipeline_ledger.state_version,
        )
        work_before = scheduler.joint_work_ledger.export_state()
        work_versions_before = (
            scheduler.joint_work_ledger.last_delivery_sequence,
            scheduler.joint_work_ledger.event_watermark,
            scheduler.joint_work_ledger.version,
            scheduler.joint_work_ledger.full_load_count,
            scheduler.joint_work_ledger.delta_load_count,
        )

        for invalid_schema_version in (3.0, True):
            bad = _empty_envelope(
                "delta", 1, base_watermark=0, base_version=0, tick=1)
            bad["schema_version"] = invalid_schema_version
            with self.subTest(schema_version=invalid_schema_version):
                with self.assertRaises(PipelineLedgerError):
                    scheduler._apply_work_ledger({"work_ledger": bad})
                self.assertIs(
                    scheduler.joint_pipeline_ledger.snapshot_state(),
                    pipeline_before)
                self.assertEqual(
                    (scheduler.joint_pipeline_ledger.delivery_sequence,
                     scheduler.joint_pipeline_ledger.watermark,
                     scheduler.joint_pipeline_ledger.state_version),
                    pipeline_versions_before)
                self.assertEqual(
                    scheduler.joint_work_ledger.export_state(), work_before)
                self.assertEqual(
                    (scheduler.joint_work_ledger.last_delivery_sequence,
                     scheduler.joint_work_ledger.event_watermark,
                     scheduler.joint_work_ledger.version,
                     scheduler.joint_work_ledger.full_load_count,
                     scheduler.joint_work_ledger.delta_load_count),
                    work_versions_before)


class V3NodeEventCanonicalRetentionTests(unittest.TestCase):
    def _two_node_frame(self):
        work_id = "ordinary#decode:1:r0n9"
        commit_payload = {
            "request_id": "ordinary", "stage": "decode",
            "generation": 1, "rank": 0, "node_id": 9,
            "operation": "compute",
            "total": {"ops": 10, "bytes": 0, "latency_ns": 1},
            "parents": [],
        }
        descriptor, frame = _full_with_descriptor(node_events=(
            _node_event(2, "commit", work_id, commit_payload),
            _node_event(3, "issue", work_id),
        ))
        return descriptor, work_id, frame

    @staticmethod
    def _typed_node_wire(row, delivery):
        mapped = dict(row["payload"])
        mapped.update({
            "kind": row["kind"],
            "work_id": row["work_id"],
            "tick": row["tick"],
            "event_id": {
                "tick": row["tick"],
                "delivery_sequence": delivery,
                "producer": "cpp_online",
                "local_seq": row["logical_seq"],
            },
        })
        return mapped

    def _assert_rejected_before_work_transaction(self, envelope):
        scheduler = object.__new__(OnlineSchedulerBase)
        scheduler.joint_pipeline_ledger = PipelineLedger()
        scheduler.joint_work_ledger = WorkLedger()
        scheduler.work_ledger_absent_seen = False
        with mock.patch.object(
                scheduler.joint_work_ledger,
                "begin_protocol_v3_transaction") as begin_transaction:
            with self.assertRaises(PipelineLedgerError):
                scheduler._apply_work_ledger({"work_ledger": envelope})
        begin_transaction.assert_not_called()
        self.assertFalse(scheduler.joint_pipeline_ledger.initialized)
        self.assertEqual(scheduler.joint_pipeline_ledger.watermark, 0)
        self.assertEqual(scheduler.joint_pipeline_ledger.state_version, 0)
        self.assertEqual(scheduler.joint_work_ledger._works, {})
        self.assertEqual(scheduler.joint_work_ledger.version, 0)
        self.assertEqual(scheduler.joint_work_ledger.last_delivery_sequence, -1)
        self.assertIsNone(scheduler.joint_work_ledger._last_event)

    def test_v3_builds_one_event_tree_and_retains_final_node_for_legacy_replay(self):
        descriptor, work_id, frame = self._two_node_frame()
        final_wire = self._typed_node_wire(frame["node_transitions"][-1], 0)
        expected_final = work_ledger_module._typed_event_from_mapping(
            final_wire).canonical_content

        # Put a valid pipeline row after both node rows: retention follows the
        # last WorkLedger node event, not the last logical event in the frame.
        parse_frame = copy.deepcopy(frame)
        parse_frame["logical_seq_end"] = 4
        parse_frame["logical_event_count"] = 4
        parse_frame["pipeline_transition_runs"].append(_pipeline_run(
            descriptor.identity.segment_id, 4, "service_done", 0))
        expected_bytes = pipeline_ledger_module._canonical_envelope_bytes(
            parse_frame)
        parser = PipelineLedger()
        event_canonical_builder = work_ledger_module._event_canonical_content
        with mock.patch.object(
                work_ledger_module, "_event_canonical_content",
                wraps=event_canonical_builder) as build_canonical:
            parsed = parser.validate_envelope(parse_frame)

        self.assertEqual(build_canonical.call_count, 1)
        self.assertIsNone(parsed.node_events[0][1].canonical_content)
        self.assertEqual(parsed.node_events[-1][1].canonical_content,
                         expected_final)
        self.assertEqual(parsed.canonical_bytes, expected_bytes)

        scheduler = object.__new__(OnlineSchedulerBase)
        scheduler.joint_pipeline_ledger = PipelineLedger()
        scheduler.joint_work_ledger = WorkLedger()
        scheduler.work_ledger_absent_seen = False
        accepted_bytes = scheduler._apply_work_ledger(
            {"work_ledger": frame})
        ledger = scheduler.joint_work_ledger
        self.assertEqual(accepted_bytes,
                         pipeline_ledger_module._canonical_envelope_bytes(
                             frame))
        self.assertEqual(ledger.work(work_id).state, "issued")
        self.assertEqual(scheduler.joint_pipeline_ledger.watermark, 3)
        self.assertEqual(ledger._last_event,
                         ("cpp_online", 3, expected_final))

        full_canonical_reference = WorkLedger()
        for row in frame["node_transitions"]:
            full_canonical_reference._apply_event(
                work_ledger_module._typed_event_from_mapping(
                    self._typed_node_wire(row, 0)))
        self.assertEqual(ledger.export_state(),
                         full_canonical_reference.export_state())
        self.assertEqual(ledger._last_event,
                         full_canonical_reference._last_event)

        accepted_state = ledger.export_state()
        accepted_version = ledger.version
        replayed_before = ledger.replayed_event_count
        duplicate = work_ledger_module._typed_event_from_mapping(final_wire)
        self.assertFalse(ledger._apply_event(duplicate))
        self.assertEqual(ledger.replayed_event_count, replayed_before + 1)
        self.assertEqual(ledger.export_state(), accepted_state)
        self.assertEqual(ledger.version, accepted_version)

        changed_wire = dict(final_wire)
        changed_wire["operation"] = "different-operation"
        altered = work_ledger_module._typed_event_from_mapping(changed_wire)
        with self.assertRaisesRegex(WorkLedgerError, "different content"):
            ledger._apply_event(altered)
        self.assertEqual(ledger.export_state(), accepted_state)
        self.assertEqual(ledger.version, accepted_version)

    def test_v3_parser_uses_scalar_factory_once_per_node_and_generic_never(self):
        for node_count in (8, 32, 128):
            with self.subTest(node_count=node_count):
                rows = tuple(
                    _node_event(
                        index + 2, "issue",
                        f"probe#decode:0:r0n{index}",
                        {"operation": "compute", "parents": []})
                    for index in range(node_count))
                _descriptor, envelope = _full_with_descriptor(
                    node_events=rows)
                expected_events = tuple(
                    work_ledger_module._typed_event_from_mapping(
                        self._typed_node_wire(row, 0))
                    for row in envelope["node_transitions"])
                expected_bytes = (
                    pipeline_ledger_module._canonical_envelope_bytes(envelope))
                scalar_factory = (
                    pipeline_ledger_module
                    ._typed_v3_node_event_from_validated_scalars)
                with mock.patch.object(
                        pipeline_ledger_module,
                        "_typed_v3_node_event_from_validated_scalars",
                        wraps=scalar_factory) as scalar_calls, \
                        mock.patch.object(
                            work_ledger_module, "_typed_event_from_mapping",
                            wraps=work_ledger_module._typed_event_from_mapping
                        ) as generic_calls, \
                        mock.patch.object(
                            work_ledger_module, "_event_canonical_content",
                            wraps=work_ledger_module._event_canonical_content
                        ) as canonical_calls:
                    parsed = PipelineLedger().validate_envelope(envelope)

                self.assertEqual(scalar_calls.call_count, node_count)
                self.assertEqual(generic_calls.call_count, 0)
                self.assertEqual(canonical_calls.call_count, 1)
                self.assertEqual(parsed.canonical_bytes, expected_bytes)
                self.assertEqual(len(parsed.node_events), node_count)
                for index, ((seq, actual), expected) in enumerate(zip(
                        parsed.node_events, expected_events)):
                    self.assertEqual(seq, expected.local_seq)
                    self.assertEqual(actual.kind, expected.kind)
                    self.assertEqual(actual.work_id, expected.work_id)
                    self.assertEqual(actual.tick, expected.tick)
                    self.assertEqual(actual.producer, expected.producer)
                    self.assertEqual(actual.delivery_sequence,
                                     expected.delivery_sequence)
                    self.assertEqual(actual.local_seq, expected.local_seq)
                    self.assertEqual(actual.delta_served,
                                     expected.delta_served)
                    self.assertEqual(
                        actual.payload,
                        {key: value for key, value in expected.payload.items()
                         if key not in ("kind", "work_id", "tick", "event_id")})
                    if index == node_count - 1:
                        self.assertEqual(actual.canonical_content,
                                         expected.canonical_content)
                    else:
                        self.assertIsNone(actual.canonical_content)

    def test_bad_node_rows_still_reject_before_either_transaction_begins(self):
        def bad_sequence_float(row):
            row["logical_seq"] = float(row["logical_seq"])

        def bad_sequence_bool(row):
            row["logical_seq"] = True

        def bad_tick_float(row):
            row["tick"] = float(row["tick"])

        def bad_tick_bool(row):
            row["tick"] = True

        def bad_kind_type(row):
            row["kind"] = 1

        def bad_kind_value(row):
            row["kind"] = "unknown-kind"

        def empty_work_id(row):
            row["work_id"] = ""

        def bad_work_id_type(row):
            row["work_id"] = 1

        def unknown_payload_key(row):
            row["payload"]["unknown_field"] = 1

        def remaining_on_wrong_kind(row):
            row["kind"] = "issue"
            row["payload"]["remaining_ops"] = 1

        def nonfinite_remaining(row):
            row["kind"] = "progress"
            row["payload"]["remaining_ops"] = float("nan")

        def bad_delta_keys(row):
            row["kind"] = "progress"
            row["payload"]["delta_served"] = {"unknown_amount": 1}

        def bad_delta_type(row):
            row["kind"] = "progress"
            row["payload"]["delta_served"] = 1

        def bad_delta_bool(row):
            row["kind"] = "progress"
            row["payload"]["delta_served"] = {"ops": True}

        def negative_delta(row):
            row["kind"] = "progress"
            row["payload"]["delta_served"] = {"ops": -1}

        def nonfinite_delta(row):
            row["kind"] = "progress"
            row["payload"]["delta_served"] = {"ops": float("nan")}

        def unsupported_nested_json(row):
            row["payload"]["total"] = {"ops": object()}

        def nonfinite_nested_json(row):
            row["payload"]["total"] = {"ops": float("nan")}

        invalidations = (
            ("float logical sequence", bad_sequence_float),
            ("bool logical sequence", bad_sequence_bool),
            ("float event tick", bad_tick_float),
            ("bool event tick", bad_tick_bool),
            ("non-string event kind", bad_kind_type),
            ("unknown event kind", bad_kind_value),
            ("malformed work id", empty_work_id),
            ("non-string work id", bad_work_id_type),
            ("unknown event field", unknown_payload_key),
            ("remaining field on wrong kind", remaining_on_wrong_kind),
            ("nonfinite remaining amount", nonfinite_remaining),
            ("unknown delta amount key", bad_delta_keys),
            ("non-mapping delta amount", bad_delta_type),
            ("bool delta amount", bad_delta_bool),
            ("negative delta amount", negative_delta),
            ("nonfinite delta amount", nonfinite_delta),
            ("unsupported nested JSON", unsupported_nested_json),
            ("nonfinite nested JSON", nonfinite_nested_json),
        )
        for node_index in (0, 1):
            for name, invalidate in invalidations:
                with self.subTest(node_index=node_index, invalidation=name):
                    _descriptor, _work_id, envelope = self._two_node_frame()
                    invalidate(envelope["node_transitions"][node_index])
                    self._assert_rejected_before_work_transaction(envelope)

    def test_negative_remaining_parses_but_prepare_rejects_atomically(self):
        _descriptor, _work_id, envelope = self._two_node_frame()
        row = envelope["node_transitions"][1]
        row["kind"] = "progress"
        row["payload"]["delta_served"] = {"ops": 0}
        row["payload"]["remaining_ops"] = -1

        # The generic legacy parser and v3 scalar factory both accept finite
        # negative remaining values at parse time; progress conservation is
        # checked later against the committed work during transaction prepare.
        generic = work_ledger_module._typed_event_from_mapping(
            self._typed_node_wire(row, envelope["delivery_sequence"]))
        parsed = PipelineLedger().validate_envelope(envelope)
        accelerated = parsed.node_events[-1][1]
        self.assertEqual(accelerated.kind, generic.kind)
        self.assertEqual(accelerated.work_id, generic.work_id)
        self.assertEqual(accelerated.tick, generic.tick)
        self.assertEqual(accelerated.producer, generic.producer)
        self.assertEqual(accelerated.delivery_sequence,
                         generic.delivery_sequence)
        self.assertEqual(accelerated.local_seq, generic.local_seq)
        self.assertEqual(accelerated.delta_served, generic.delta_served)
        generic_body = {
            key: value for key, value in generic.payload.items()
            if key not in ("kind", "work_id", "tick", "event_id")}
        self.assertEqual(accelerated.payload, generic_body)
        self.assertEqual(accelerated.canonical_content,
                         generic.canonical_content)

        scheduler = object.__new__(OnlineSchedulerBase)
        scheduler.joint_pipeline_ledger = PipelineLedger()
        scheduler.joint_work_ledger = WorkLedger()
        scheduler.work_ledger_absent_seen = False
        pipeline = scheduler.joint_pipeline_ledger
        work = scheduler.joint_work_ledger
        pipeline_before = pipeline.snapshot_state()
        work_before = work.snapshot_state()
        pipeline_marks = (
            pipeline.initialized, pipeline.delivery_sequence,
            pipeline.watermark, pipeline.state_version)
        work_marks = (
            work.last_delivery_sequence, work.event_watermark,
            work.version, work.full_load_count, work.delta_load_count)

        with self.assertRaisesRegex(
                WorkLedgerError, "remaining_ops disagrees with delta"):
            scheduler._apply_work_ledger({"work_ledger": envelope})

        self.assertIs(pipeline.snapshot_state(), pipeline_before)
        self.assertIs(work.snapshot_state(), work_before)
        self.assertEqual(
            (pipeline.initialized, pipeline.delivery_sequence,
             pipeline.watermark, pipeline.state_version),
            pipeline_marks)
        self.assertEqual(
            (work.last_delivery_sequence, work.event_watermark,
             work.version, work.full_load_count, work.delta_load_count),
            work_marks)
        self.assertEqual(work._works, {})
        self.assertIsNone(work._last_event)

    def test_bad_outer_integer_fields_reject_before_work_transaction(self):
        invalid_fields = (
            ("delivery_sequence", 0.0),
            ("delivery_sequence", True),
            ("tick", 0.0),
            ("tick", True),
        )
        for field, value in invalid_fields:
            with self.subTest(field=field, value=value):
                _descriptor, _work_id, envelope = self._two_node_frame()
                envelope[field] = value
                self._assert_rejected_before_work_transaction(envelope)

    def test_float_and_int_payloads_remain_distinct_in_frame_replay_bytes(self):
        _descriptor, _work_id, integer_frame = self._two_node_frame()
        float_frame = copy.deepcopy(integer_frame)
        float_frame["node_transitions"][0]["payload"]["total"]["ops"] = 10.0
        integer_parsed = PipelineLedger().validate_envelope(integer_frame)
        float_parsed = PipelineLedger().validate_envelope(float_frame)
        self.assertIsNone(integer_parsed.node_events[0][1].canonical_content)
        self.assertIsNone(float_parsed.node_events[0][1].canonical_content)
        self.assertNotEqual(integer_parsed.canonical_bytes,
                            float_parsed.canonical_bytes)
