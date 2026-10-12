"""Bounded builder for algebraic remote-read prediction descriptors.

The public object produced here is the shared :class:`PipelineDescriptor`.
The small mutable draft only exists while an action graph is being assembled;
tile IDs and registration positions are ranges, never per-tile Python objects.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable, Optional

from joint.pipeline_descriptor import PipelineDescriptor, PipelineIdentity


class VirtualLegIds(Sequence):
    """Lazy legacy ID range supporting bounded indexing and iteration."""

    __slots__ = ("prefix", "first_index", "count")

    def __init__(self, prefix: str, first_index: int, count: int):
        if not prefix or type(first_index) is not int or first_index < 0:
            raise ValueError("invalid virtual leg ID range")
        if type(count) is not int or count < 0:
            raise ValueError("invalid virtual leg ID count")
        self.prefix = prefix
        self.first_index = first_index
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self.count)
            return tuple(self.prefix + str(self.first_index + i)
                         for i in range(start, stop, step))
        if type(index) is not int:
            raise TypeError("virtual leg ID index must be an integer")
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError(index)
        return self.prefix + str(self.first_index + index)


class PhysicalPipelineDraft:
    """Bounded affine emitter for one physical owner-domain descriptor.

    The API accepts extents and endpoint templates, not individual transfer
    tiles.  A caller splits only at source/pass/body/legacy-emission
    boundaries, reserves the real global tag span once, then supplies the
    rank-local affine mappings.  Retained storage is proportional to those
    structural runs and bounded physical anchors.
    """

    def __init__(self, *, identity: PipelineIdentity, stream_kind: str,
                 projection_chains=()):
        if stream_kind not in ("prefill_read_window", "decode_remote_credit"):
            raise ValueError(f"unsupported physical stream kind {stream_kind!r}")
        expected = "prefill" if stream_kind == "prefill_read_window" else "decode"
        if identity.stage != expected:
            raise ValueError("physical stream kind disagrees with identity stage")
        self.identity = identity
        self.stream_kind = stream_kind
        self.groups = []
        self.tag_allocation_runs = []
        self.endpoint_runs = []
        self.endpoint_dependency_runs = []
        self.consume_completion_runs = []
        self.anchors = []
        self._holes = []
        self._projection_chains = list(projection_chains)
        self._anchor_roles = set()
        self._tag_allocations = {}

    def add_group_range(self, *, pass_or_window_ordinal: int,
                        source_ordinal: int, source_key: str,
                        layer_start: int, layer_end: int,
                        global_tile_first: int, tile_count: int,
                        axis_unit: str, total_units: int,
                        units_per_tile: int, tail_units: int,
                        owner_shards, shard_total_bytes,
                        shard_bytes_per_unit=None) -> int:
        """Add one rank-local source/pass affine group, with no tile loop.

        ``owner_shards`` is the original-order subset for this descriptor's
        target rank. ``shard_total_bytes`` is the exact aggregate byte total
        for each corresponding shard across this active extent. ``axis_unit``
        and ``shard_bytes_per_unit`` preserve whether the frozen planner used
        a shared byte-stream axis or source-token units with per-shard byte
        ratios. Ranks whose
        shard stream ends earlier get a shorter group; zero-work padded tiles
        are forbidden.
        """
        owner_shards = tuple(owner_shards)
        shard_total_bytes = tuple(shard_total_bytes)
        if not owner_shards or len(owner_shards) != len(shard_total_bytes):
            raise ValueError("owner shard metadata must be a nonempty aligned tuple")
        def field(shard, name):
            return shard[name] if isinstance(shard, dict) else getattr(shard, name)

        if any(int(field(shard, "target_rank")) != self.identity.rank
               for shard in owner_shards):
            raise ValueError("group shards must target this buffer owner rank")
        if axis_unit not in ("stream_bytes", "source_tokens"):
            raise ValueError("axis_unit must be stream_bytes or source_tokens")
        if type(tile_count) is not int or tile_count <= 0:
            raise ValueError("group tile_count must be positive")
        if type(total_units) is not int or total_units <= 0:
            raise ValueError("group total_units must be positive")
        if type(units_per_tile) is not int or units_per_tile <= 0:
            raise ValueError("units_per_tile must be positive")
        if type(tail_units) is not int or not 0 < tail_units <= units_per_tile:
            raise ValueError("tail_units must be in (0, units_per_tile]")
        if total_units != (tile_count - 1) * units_per_tile + tail_units:
            raise ValueError("tile axis units/count/tail are inconsistent")
        if any(type(value) is not int or value < 0
               for value in shard_total_bytes):
            raise ValueError("shard totals must be nonnegative integers")
        if shard_bytes_per_unit is None:
            shard_bytes_per_unit = (1,) * len(owner_shards)
        shard_bytes_per_unit = tuple(shard_bytes_per_unit)
        if len(shard_bytes_per_unit) != len(owner_shards):
            raise ValueError("per-shard unit ratios do not align with shards")
        if any(type(value) is not int or value < 0
               for value in shard_bytes_per_unit):
            raise ValueError("per-shard bytes_per_unit must be nonnegative integers")
        if axis_unit == "source_tokens":
            if any(total_units * ratio != total_bytes
                   for ratio, total_bytes in zip(shard_bytes_per_unit,
                                                 shard_total_bytes)):
                raise ValueError(
                    "source-token shard totals must equal units*bytes_per_unit")
        elif (any(ratio != 1 for ratio in shard_bytes_per_unit)
              or max(shard_total_bytes) != total_units
              or any(total_bytes > total_units
                     for total_bytes in shard_total_bytes)):
            raise ValueError(
                "stream-byte shard totals must fit the shared byte axis")
        if type(global_tile_first) is not int or global_tile_first <= 0:
            raise ValueError("global_tile_first must be positive")
        if type(layer_start) is not int or type(layer_end) is not int or layer_start < 0 or layer_end <= layer_start:
            raise ValueError("group layer range must be nonempty")
        group_ordinal = len(self.groups)
        shards = []
        for ordinal, (shard, total_bytes, bytes_per_unit) in enumerate(
                zip(owner_shards, shard_total_bytes, shard_bytes_per_unit)):
            shards.append({
                "shard_ordinal": ordinal,
                "source_rank": int(field(shard, "source_rank")),
                "target_rank": int(field(shard, "target_rank")),
                "total_data_bytes": int(total_bytes),
                "bytes_per_unit": bytes_per_unit,
                "noc_path": [int(rank) for rank in field(shard, "noc_path")],
            })
        self.groups.append({
            "group_ordinal": group_ordinal,
            "pass_or_window_ordinal": int(pass_or_window_ordinal),
            "source_ordinal": int(source_ordinal),
            "source_key": str(source_key),
            "layer_start": layer_start,
            "layer_end": layer_end,
            "tile_axis": {
                "unit": axis_unit,
                "total_units": total_units,
                "units_per_tile": units_per_tile,
                "tile_count": tile_count,
                "tail_units": tail_units,
                "global_tile_first": global_tile_first,
            },
            "shards": shards,
        })
        return group_ordinal

    @staticmethod
    def _copy_endpoint_run(run):
        required = {
            "rank", "group_ordinal", "first_tile", "tile_count",
            "shard_ordinals", "first_json_id", "tile_stride",
            "shard_stride", "initial_parent_json_id", "legacy_context",
            "legs",
        }
        if not isinstance(run, dict) or set(run) != required:
            raise ValueError("endpoint run must have the exact typed wire keys")
        copied = dict(run)
        copied["shard_ordinals"] = list(run["shard_ordinals"])
        copied["legacy_context"] = dict(run["legacy_context"])
        copied["legs"] = [dict(leg) for leg in run["legs"]]
        return copied

    def add_tag_allocation_reference(self, *, allocator_scope_id: str,
                                     emission_ordinal: int, first_tag: int,
                                     count: int, group_ordinal: int,
                                     first_tile: int, tile_count: int,
                                     first_shard: int, shard_count: int) -> None:
        """Reference one real bulk allocator reservation from this owner.

        Several owner descriptors may refer to the same global allocation;
        only the owner-local group/tile/shard mapping differs.  The allocation
        identity tuple must match exactly when repeated within a draft.
        """
        allocation = (str(allocator_scope_id), int(emission_ordinal),
                      int(first_tag), int(count))
        prior = self._tag_allocations.get(int(emission_ordinal))
        if prior is not None and prior != allocation:
            raise ValueError("tag emission ordinal changed allocation identity")
        if prior is not None:
            raise ValueError(
                "one owner descriptor must reference an allocation once; "
                "split a new range at the next real emission point")
        self._tag_allocations[int(emission_ordinal)] = allocation
        self.tag_allocation_runs.append({
            "allocator_scope_id": allocation[0],
            "emission_ordinal": allocation[1],
            "first_tag": allocation[2],
            "count": allocation[3],
            "group_ordinal": int(group_ordinal),
            "first_tile": int(first_tile),
            "first_shard": int(first_shard),
            "tile_count": int(tile_count),
            "shard_count": int(shard_count),
        })

    def _append_hole_progression(self, *, rank: int, first_json_id: int,
                                 tile_count: int, tile_stride: int,
                                 excluded_indices=()) -> None:
        """Store an AP minus a bounded anchor set as O(anchor count) APs."""
        excluded = sorted(set(int(index) for index in excluded_indices))
        if any(index < 0 or index >= tile_count for index in excluded):
            raise ValueError("hole exception lies outside the endpoint range")
        cursor = 0
        for index in excluded:
            if index > cursor:
                self._holes.append({
                    "rank": int(rank),
                    "first_json_id": int(first_json_id) + cursor * int(tile_stride),
                    "count": index - cursor,
                    "stride": int(tile_stride),
                    "kind": "virtual_endpoint_hole",
                })
            cursor = index + 1
        if cursor < tile_count:
            self._holes.append({
                "rank": int(rank),
                "first_json_id": int(first_json_id) + cursor * int(tile_stride),
                "count": tile_count - cursor,
                "stride": int(tile_stride),
                "kind": "virtual_endpoint_hole",
            })

    def add_endpoint_run(self, run, *, anchor_exceptions=()) -> int:
        """Append a tile/shard affine endpoint template, never per-tile rows.

        ``anchor_exceptions`` contains only bounded structural slots, each as
        ``(tile_ordinal, shard_ordinal, endpoint, role)``.  All other IDs in
        each leg's tile progression become virtual holes.
        """
        copied = self._copy_endpoint_run(run)
        group = self._group(copied["group_ordinal"])
        first_tile = copied["first_tile"]
        tile_count = copied["tile_count"]
        if first_tile < 1 or first_tile + tile_count - 1 > group["tile_axis"]["tile_count"]:
            raise ValueError("endpoint range is outside its group tile axis")
        shard_ordinals = tuple(copied["shard_ordinals"])
        if not shard_ordinals or tuple(sorted(set(shard_ordinals))) != shard_ordinals:
            raise ValueError("endpoint shard ordinals must be sorted and unique")
        if any(shard < 0 or shard >= len(group["shards"])
               for shard in shard_ordinals):
            raise ValueError("endpoint shard ordinal is outside its group")
        self._validate_legacy_context(copied["legacy_context"])
        run_index = len(self.endpoint_runs)
        self.endpoint_runs.append(copied)
        exceptions = {}
        for tile, shard, endpoint, role in anchor_exceptions:
            if not (first_tile <= int(tile) < first_tile + tile_count):
                raise ValueError("anchor exception tile is outside endpoint run")
            if int(shard) not in shard_ordinals:
                raise ValueError("anchor exception shard is outside endpoint run")
            if endpoint not in {leg["endpoint"] for leg in copied["legs"]}:
                raise ValueError("anchor exception endpoint is absent from endpoint run")
            exceptions.setdefault((int(shard), endpoint), []).append(
                (int(tile) - first_tile, role))
        for local_shard, shard in enumerate(shard_ordinals):
            for leg in copied["legs"]:
                first_id = (copied["first_json_id"]
                            + local_shard * copied["shard_stride"]
                            + leg["json_offset"])
                selected = exceptions.get((shard, leg["endpoint"]), ())
                excluded = [index for index, _role in selected]
                self._append_hole_progression(
                    rank=copied["rank"], first_json_id=first_id,
                    tile_count=tile_count, tile_stride=copied["tile_stride"],
                    excluded_indices=excluded)
                for local_tile, role in selected:
                    self._append_physical_anchor(
                        role=role, rank=copied["rank"],
                        run_index=run_index,
                        tile_ordinal=first_tile + local_tile,
                        shard_ordinal=shard,
                        endpoint=leg["endpoint"])
        return run_index

    @staticmethod
    def _validate_legacy_context(context):
        if not isinstance(context, dict) or set(context) != {
                "request_id", "stage", "generation"}:
            raise ValueError("legacy_context has invalid exact keys")
        stage = context["stage"]
        if stage not in ("prefill", "decode", "completion"):
            raise ValueError("legacy_context stage is invalid")
        generation = 0 if stage == "prefill" else 1
        if type(context["generation"]) is not int or context["generation"] != generation:
            raise ValueError("legacy_context generation disagrees with stage")
        if not isinstance(context["request_id"], str) or not context["request_id"]:
            raise ValueError("legacy_context request_id must be nonempty")

    def add_endpoint_dependency_run(self, run) -> None:
        required = {"rank", "group_ordinal", "first_tile", "tile_count",
                    "shard_ordinals", "endpoint", "dependency_kind",
                    "first_dependency_json_id", "dependency_tile_stride",
                    "dependency_shard_stride"}
        if not isinstance(run, dict) or set(run) != required:
            raise ValueError("endpoint dependency run has invalid exact keys")
        copied = dict(run)
        copied["shard_ordinals"] = list(run["shard_ordinals"])
        self.endpoint_dependency_runs.append(copied)

    def add_consume_completion_run(self, run) -> None:
        required = {"rank", "group_ordinal", "first_tile", "tile_count",
                    "shard_ordinals", "dependency_kind",
                    "first_dependency_json_id", "dependency_tile_stride",
                    "dependency_shard_stride"}
        if not isinstance(run, dict) or set(run) != required:
            raise ValueError("consume completion run has invalid exact keys")
        copied = dict(run)
        copied["shard_ordinals"] = list(run["shard_ordinals"])
        self.consume_completion_runs.append(copied)

    def set_projection_chains(self, chains) -> None:
        """Attach the same typed projection chains used by the candidate model."""
        self._projection_chains = [dict(chain) for chain in chains]

    def _group(self, group_ordinal: int):
        if type(group_ordinal) is not int or not 0 <= group_ordinal < len(self.groups):
            raise ValueError("unknown physical group ordinal")
        return self.groups[group_ordinal]

    def _append_physical_anchor(self, *, role, rank, run_index, tile_ordinal,
                                shard_ordinal, endpoint):
        if role not in ("start", "frontier"):
            raise ValueError("physical anchors only support start/frontier roles")
        run = self.endpoint_runs[run_index]
        local_tile = int(tile_ordinal) - run["first_tile"]
        local_shard = run["shard_ordinals"].index(int(shard_ordinal))
        leg = next(item for item in run["legs"] if item["endpoint"] == endpoint)
        node_id = (run["first_json_id"] + local_tile * run["tile_stride"]
                   + local_shard * run["shard_stride"] + leg["json_offset"])
        if endpoint == "body_marker":
            selector = {
                "kind": "internal_marker_done",
                "group_ordinal": run["group_ordinal"],
                "tile_ordinal": int(tile_ordinal),
                "shard_ordinal": int(shard_ordinal),
                "endpoint": "body_marker",
            }
        else:
            if leg["tag_emission_ordinal"] is None:
                raise ValueError("physical tagged endpoint anchor lacks tag allocation")
            tag_offset = (leg["tag_offset"] + local_tile * leg["tag_tile_stride"]
                          + local_shard * leg["tag_shard_stride"])
            selector = {
                "kind": "tile_endpoint_leg",
                "group_ordinal": run["group_ordinal"],
                "tile_ordinal": int(tile_ordinal),
                "shard_ordinal": int(shard_ordinal),
                "endpoint": endpoint,
                "tag_allocation_emission_ordinal": leg["tag_emission_ordinal"],
                "tag_offset": tag_offset,
            }
        self.anchors.append({
            "role": role, "rank": int(rank),
            "binding": "physical_endpoint_slot", "node_id": node_id,
            "selector": selector,
        })
        self._anchor_roles.add(role)

    def _coalesce_affine_runs(self) -> None:
        """Merge per-emission endpoint runs into maximal wide tile ranges.

        Deferred buffer-reuse releases emit one bounded extent per real
        release point (typically one tile), each with its own transfer-tag
        reservation.  A long train therefore accumulates thousands of
        single-tile ``endpoint_runs`` that describe one physical affine
        stream.  This pass coalesces them back into wide runs after all
        emission points are known, keeping the represented physics
        byte-for-byte identical:

        * Endpoint identity.  A survivor absorbs its successor only when
          the successor's ``first_json_id`` is exactly the survivor's
          affine continuation (``A.first_json_id + delta_tiles *
          A.tile_stride``) with identical rank/group/shards/legs/strides.
          The merged run's (tile, shard, leg) -> json_id map is therefore
          the exact concatenation of the fragmented maps: every endpoint
          position, its ID, and its position order inside the run walk
          (tile-major, shard, leg) are unchanged.
        * Dependencies.  ``endpoint_dependency_runs`` key parents by
          (rank, group, endpoint, tile, shard) and are never rewritten;
          merging run boundaries cannot change any parent edge, so no
          merge crosses a dependency boundary by construction.
        * Tags.  Per-emission tag allocations are absorbed only when the
          allocator handed out globally contiguous spans
          (``B.first_tag == A.first_tag + A.count``) for tile-adjacent
          extents of the same group.  The survivor entry keeps its
          emission ordinal and original ``first_tag``; absorbed legs and
          anchor selectors are rebased by the contiguity delta, so every
          (tile, shard, endpoint) still resolves to the identical tag
          value in every descriptor that shares the allocation.
        * Anchors.  Physical anchor selectors keep their (tile, shard,
          endpoint) and node_id; their tag fields are recomputed from the
          surviving run's leg formula, which reproduces the original
          value whenever nothing merged.
        * Cursor semantics.  The executor walks each run with one cursor
          in tile-major order and looks up dependencies/protocol state by
          absolute tile ordinal, so a wide run replays the identical
          transfer sequence; per-position readiness gates, slot claims,
          and consume completions are all keyed by absolute tile.
        """
        # Phase 1: absorb tile-adjacent tag allocations with contiguous
        # global tag spans into the earlier entry of each group.
        absorbed = {}
        tag_entries_by_group = {}
        entry_position = {id(entry): index for index, entry
                          in enumerate(self.tag_allocation_runs)}
        for entry in self.tag_allocation_runs:
            tag_entries_by_group.setdefault(
                int(entry["group_ordinal"]), []).append(entry)
        merged_tag_runs = []
        for group_ordinal in sorted(tag_entries_by_group):
            entries = sorted(tag_entries_by_group[group_ordinal],
                             key=lambda item: int(item["first_tile"]))
            survivor = None
            for entry in entries:
                if (survivor is not None
                        and entry["allocator_scope_id"]
                        == survivor["allocator_scope_id"]
                        and int(entry["first_tile"]) == int(
                            survivor["first_tile"])
                        + int(survivor["tile_count"])
                        and int(entry["first_tag"]) == int(
                            survivor["first_tag"]) + int(survivor["count"])):
                    survivor["count"] = (int(survivor["count"])
                                         + int(entry["count"]))
                    survivor["tile_count"] = (int(survivor["tile_count"])
                                              + int(entry["tile_count"]))
                    shard_end = max(
                        int(survivor["first_shard"])
                        + int(survivor["shard_count"]),
                        int(entry["first_shard"]) + int(entry["shard_count"]))
                    survivor["first_shard"] = min(
                        int(survivor["first_shard"]),
                        int(entry["first_shard"]))
                    survivor["shard_count"] = (
                        shard_end - int(survivor["first_shard"]))
                    absorbed[int(entry["emission_ordinal"])] = (
                        int(survivor["emission_ordinal"]),
                        int(entry["first_tag"]) - int(survivor["first_tag"]))
                    continue
                survivor = dict(entry)
                merged_tag_runs.append((entry_position[id(entry)], survivor))
        # 合并只做吸收，survivors 的相对顺序必须保持原序列序——按组
        # 聚集输出会重组跨 group 交错的同一 allocator scope，破坏
        # "emission_ordinal 严格递增且 tag 连续不减"校验（序列化顺序
        # 问题，不改变任何分配内容）。
        merged_tag_runs.sort(key=lambda pair: pair[0])
        self.tag_allocation_runs = [item for _, item in merged_tag_runs]

        # Phase 2: rebase absorbed emissions inside every endpoint leg.
        for run in self.endpoint_runs:
            for leg in run["legs"]:
                if leg["tag_emission_ordinal"] is None:
                    continue
                absorption = absorbed.get(int(leg["tag_emission_ordinal"]))
                if absorption is None:
                    continue
                survivor_ordinal, tag_delta = absorption
                leg["tag_emission_ordinal"] = survivor_ordinal
                leg["tag_offset"] = int(leg["tag_offset"]) + tag_delta

        # Phase 3: merge tile-adjacent endpoint runs whose ID and tag
        # mappings are exact affine continuations.
        def _merge_key(run):
            context = run["legacy_context"]
            return (int(run["rank"]), int(run["group_ordinal"]),
                    tuple(run["shard_ordinals"]), int(run["tile_stride"]),
                    int(run["shard_stride"]),
                    (context["request_id"], context["stage"],
                     context["generation"]),
                    tuple((leg["endpoint"], int(leg["json_offset"]),
                           leg["tag_emission_ordinal"],
                           leg["tag_tile_stride"], leg["tag_shard_stride"])
                          for leg in run["legs"]))

        chains = {}
        for run in self.endpoint_runs:
            chains.setdefault(_merge_key(run), []).append(run)
        absorbed_runs = set()
        for members in chains.values():
            if len(members) < 2:
                continue
            members.sort(key=lambda item: int(item["first_tile"]))
            head = members[0]
            for successor in members[1:]:
                delta_tiles = (int(successor["first_tile"])
                               - int(head["first_tile"])
                               - int(head["tile_count"]))
                id_contiguous = (
                    int(successor["first_json_id"]) == int(
                        head["first_json_id"])
                    + (int(successor["first_tile"])
                       - int(head["first_tile"])) * int(head["tile_stride"]))
                tags_contiguous = all(
                    (head_leg["tag_offset"] is None
                     and successor_leg["tag_offset"] is None)
                    or (head_leg["tag_offset"] is not None
                        and successor_leg["tag_offset"] is not None
                        and int(successor_leg["tag_offset"]) == int(
                            head_leg["tag_offset"])
                        + (int(successor["first_tile"])
                           - int(head["first_tile"]))
                        * int(head_leg["tag_tile_stride"]))
                    for head_leg, successor_leg in zip(head["legs"],
                                                       successor["legs"]))
                if delta_tiles == 0 and id_contiguous and tags_contiguous:
                    head["tile_count"] = (int(head["tile_count"])
                                          + int(successor["tile_count"]))
                    absorbed_runs.add(id(successor))
        if absorbed_runs:
            self.endpoint_runs = [run for run in self.endpoint_runs
                                  if id(run) not in absorbed_runs]

        # Phase 4: recompute physical anchor tag selectors from the
        # surviving runs (reproduces the original value when unmerged).
        for anchor in self.anchors:
            selector = anchor["selector"]
            if selector.get("kind") != "tile_endpoint_leg":
                continue
            tile_ordinal = int(selector["tile_ordinal"])
            shard_ordinal = int(selector["shard_ordinal"])
            for run in self.endpoint_runs:
                if (int(run["group_ordinal"])
                        != int(selector["group_ordinal"])
                        or not int(run["first_tile"]) <= tile_ordinal
                        < int(run["first_tile"]) + int(run["tile_count"])
                        or shard_ordinal not in run["shard_ordinals"]):
                    continue
                leg = next((item for item in run["legs"]
                            if item["endpoint"] == selector["endpoint"]),
                           None)
                if leg is None or leg["tag_emission_ordinal"] is None:
                    continue
                local_tile = tile_ordinal - int(run["first_tile"])
                local_shard = run["shard_ordinals"].index(shard_ordinal)
                selector["tag_allocation_emission_ordinal"] = (
                    leg["tag_emission_ordinal"])
                selector["tag_offset"] = (
                    int(leg["tag_offset"])
                    + local_tile * int(leg["tag_tile_stride"])
                    + local_shard * int(leg["tag_shard_stride"]))
                break

    def finish(self) -> PipelineDescriptor:
        if not self.groups or not self.endpoint_runs:
            raise ValueError("physical pipeline descriptor has no emitted endpoint")
        if not {"start", "frontier"}.issubset(self._anchor_roles):
            raise ValueError("physical descriptor needs bounded start/frontier anchors")
        self._coalesce_affine_runs()
        identity = self.identity
        anchors = [*self.anchors,
            {"role": "service", "rank": identity.rank,
             "binding": "descriptor_cursor_selector", "node_id": None,
             "selector": {"kind": "existing_service_gate",
                          "existing_gate": "remote_read_service_gate"}},
            {"role": "settle", "rank": identity.rank,
             "binding": "descriptor_cursor_selector", "node_id": None,
             "selector": {"kind": "all_obligations_settled"}},
        ]
        self._holes.sort(key=lambda item: (item["rank"], item["first_json_id"]))
        rank = identity.rank
        wire = {
            "descriptor_version": 1,
            "identity": {
                "request_id": identity.request_id,
                "stage": identity.stage,
                "generation": identity.generation,
                "stage_epoch": identity.stage_epoch,
                "segment_ordinal": identity.segment_ordinal,
                "rank": rank,
                "staging_allocation_id": identity.staging_allocation_id,
                "segment_id": identity.segment_id,
                "buffer_domain_id": identity.buffer_domain_id,
            },
            "stream_kind": self.stream_kind,
            "groups": self.groups,
            "protocol": {
                "buffer_slots": 2,
                "release_lag_tiles": 2,
                "initial_tiles_without_release": 2,
                "fill_ack_bytes": 1,
                "release_ack_bytes": 1,
                "hbm_charge": True,
                "target_rank_order": "release_ack_send_i_if_required_then_data_recv_i_then_fill_ack_send_i_then_data_recv_i_plus_1",
                "source_rank_order": "release_ack_recv_i_if_required_then_data_send_i_then_fill_ack_recv_i",
                "tile_branching": "tile1_head_main_chain_tile2_immediate_side_branch_tile3_plus_release_gated",
            },
            "prediction_profile": {
                "prefill_consume": "zero_duration_no_extra_consume",
                "decode_consume": {
                    "kind": "profiled_consume",
                    "profile_id": "decode-service-profile-v1",
                    "compute_ns_total": 0.0,
                    "hbm_bytes_total": 0,
                    "resource_key": f"hbm:{rank}",
                    "share_rule": "existing_action_work_graph_byte_share_vector",
                },
            },
            "execution_profile": {
                "data_legs": "ordinary_comm_send_recv_path",
                "endpoint_hbm_rule": "one_original_COMM_READ_or_COMM_WRITE_job_per_charged_endpoint",
                "prefill_consume": "existing_body_or_marker_only",
                "decode_consume": "existing_body_nodes_only",
            },
            "logical_id_ranges": self._holes,
            "tag_allocation_runs": self.tag_allocation_runs,
            "endpoint_runs": self.endpoint_runs,
            "endpoint_dependency_runs": self.endpoint_dependency_runs,
            "consume_completion_runs": self.consume_completion_runs,
            "projection": {"version": 1, "chains": self._projection_chains},
            "anchors": anchors,
        }
        return PipelineDescriptor.from_wire(wire)


class CompactPipelinePlan:
    """Build one owner-rank descriptor using group and affine ID ranges."""

    def __init__(self, *, work_id: str, tag: str,
                 identity: PipelineIdentity, stream_kind: str,
                 reserve_ordinals: Callable[[int], int], gate_id: str,
                 legacy_rank_index: int,
                 prefill_consume: str = "zero_duration_no_extra_consume"):
        if stream_kind not in ("prefill_read_window", "decode_remote_credit"):
            raise ValueError(f"unsupported remote-read stream kind {stream_kind!r}")
        if identity.stage != ("prefill" if stream_kind == "prefill_read_window" else "decode"):
            raise ValueError("pipeline identity stage disagrees with stream kind")
        self.work_id = work_id
        self.tag = tag
        self.identity = identity
        self.stream_kind = stream_kind
        self.reserve_ordinals = reserve_ordinals
        self.gate_id = gate_id
        if type(legacy_rank_index) is not int or legacy_rank_index < 0:
            raise ValueError("legacy rank index must be a non-negative integer")
        self.legacy_rank_index = legacy_rank_index
        self.prefill_consume = prefill_consume
        self.groups = []
        self.chains = []
        self._next_index = 0
        self._next_global_tile = 1
        self._previous_fill = None
        self._previous_consume = None
        self._decode_profile = None
        self._resource_key = f"hbm:{identity.rank}"

    def begin_group(self, *, pass_or_window_ordinal: int,
                    source_ordinal: int, source_key: str,
                    layer_start: int, layer_end: int,
                    total_units: int, units_per_tile: int,
                    tile_count: int, tail_units: int,
                    source_rank: int, target_rank: int,
                    noc_path, route, latency_ns: int,
                    start_deps=()):
        if target_rank != self.identity.rank:
            raise ValueError("pipeline group target rank must own its buffer domain")
        if min(total_units, units_per_tile, tile_count, tail_units) <= 0:
            raise ValueError("compact pipeline group dimensions must be positive")
        if (tile_count != (total_units + units_per_tile - 1) // units_per_tile
                or tail_units != total_units - (tile_count - 1) * units_per_tile):
            raise ValueError("compact pipeline tile axis is inconsistent")
        group_ordinal = len(self.groups)
        first_index = self._next_index
        fill_prefix = f"{self.work_id}#{self.tag}#R{self.legacy_rank_index}#F"
        fill_order_first = self.reserve_ordinals(tile_count)
        # The legacy rank-local fill list is one serial chain spanning every
        # source/pass group: only its first fill has the external start deps;
        # later fills depend solely on the preceding fill. Do not retain the
        # start barrier on every later source boundary.
        deps = (list(start_deps) if self._previous_fill is None
                else [self._previous_fill])
        group = {
            "group_ordinal": group_ordinal,
            "pass_or_window_ordinal": int(pass_or_window_ordinal),
            "source_ordinal": int(source_ordinal),
            "source_key": str(source_key),
            "layer_start": int(layer_start),
            "layer_end": int(layer_end),
            "tile_axis": {
                "unit": "stream_bytes",
                "total_units": int(total_units),
                "units_per_tile": int(units_per_tile),
                "tile_count": int(tile_count),
                "tail_units": int(tail_units),
                "global_tile_first": self._next_global_tile,
            },
            "shards": [{
                "shard_ordinal": 0,
                "source_rank": int(source_rank),
                "target_rank": int(target_rank),
                "total_data_bytes": int(total_units),
                "bytes_per_unit": 1,
                "noc_path": [int(rank) for rank in noc_path],
            }],
        }
        self.groups.append(group)
        fill = {
            "leg_id_prefix": fill_prefix,
            "first_index": first_index,
            "fill_order_first": fill_order_first,
            "fill_order_stride": 1,
            "completion_order_first": None,
            "completion_order_stride": 0,
            "route": list(route),
            "bytes_per_tile": int(units_per_tile),
            "latency_ns": int(latency_ns),
            "start_deps": deps,
            "gate_id": self.gate_id,
            "gate_capacity": 2,
        }
        self.chains.append({
            "group_ordinal": group_ordinal,
            "rank": target_rank,
            "tile_count": tile_count,
            "fill": fill,
            "consume": None,
        })
        self._next_index += tile_count
        self._next_global_tile += tile_count
        self._previous_fill = fill_prefix + str(first_index + tile_count - 1)
        self._resource_key = f"hbm:{target_rank}"
        return len(self.chains) - 1, VirtualLegIds(fill_prefix, first_index, tile_count)

    def add_consume(self, chain_handle: int, *, kind: str,
                    route=(), group: Optional[str] = None,
                    latency_first_ns: int = 0, latency_rest_ns: int = 0,
                    consume_deps=(), compute_share=None, hbm_share=None):
        chain = self.chains[chain_handle]
        if chain["consume"] is not None:
            raise ValueError("compact pipeline chain already has a consume")
        count = chain["tile_count"]
        first_index = chain["fill"]["first_index"]
        prefix = f"{self.work_id}#{self.tag}#R{self.legacy_rank_index}#C"
        consume_order_first = self.reserve_ordinals(count)
        start_deps = []
        if self._previous_consume is not None:
            start_deps.append(self._previous_consume)
        chain["consume"] = {
            "leg_id_prefix": prefix,
            "first_index": first_index,
            "consume_order_first": consume_order_first,
            "consume_order_stride": 1,
            "completion_order_first": None,
            "completion_order_stride": 0,
            "kind": kind,
            "route": list(route),
            "group": group,
            "latency_first_ns": int(latency_first_ns),
            "latency_rest_ns": int(latency_rest_ns),
            "consume_deps": list(consume_deps),
            "consume_start_deps": start_deps,
            "gate_id": self.gate_id,
            "compute_share": compute_share,
            "hbm_share": hbm_share,
        }
        self._previous_consume = prefix + str(first_index + count - 1)
        if kind == "compute":
            self._decode_profile = (compute_share, hbm_share, route)
            self._resource_key = route[0]
        return VirtualLegIds(prefix, first_index, count)

    def terminal_ids(self) -> tuple[str, ...]:
        """Bounded consume terminals for service/settle dependency folding."""
        terminals = []
        for chain in self.chains:
            consume = chain["consume"]
            if consume is None:
                continue
            terminals.append(
                consume["leg_id_prefix"]
                + str(consume["first_index"] + chain["tile_count"] - 1))
        return tuple(terminals)

    def finish(self) -> PipelineDescriptor:
        if not self.chains or any(chain["consume"] is None for chain in self.chains):
            raise ValueError("compact pipeline contains an unfinished chain")
        if self._decode_profile is None:
            compute_total = 0.0
            hbm_total = 0
        else:
            compute_share, hbm_share, _route = self._decode_profile
            compute_total = float(compute_share["total_units"])
            hbm_total = int(hbm_share["total_units"])
        rank = self.identity.rank
        wire = {
            "descriptor_version": 1,
            "identity": {
                "request_id": self.identity.request_id,
                "stage": self.identity.stage,
                "generation": self.identity.generation,
                "stage_epoch": self.identity.stage_epoch,
                "segment_ordinal": self.identity.segment_ordinal,
                "rank": rank,
                "staging_allocation_id": self.identity.staging_allocation_id,
                "segment_id": self.identity.segment_id,
                "buffer_domain_id": self.identity.buffer_domain_id,
            },
            "stream_kind": self.stream_kind,
            "groups": self.groups,
            "protocol": {
                "buffer_slots": 2,
                "release_lag_tiles": 2,
                "initial_tiles_without_release": 2,
                "fill_ack_bytes": 1,
                "release_ack_bytes": 1,
                "hbm_charge": True,
                "target_rank_order": "release_ack_send_i_if_required_then_data_recv_i_then_fill_ack_send_i_then_data_recv_i_plus_1",
                "source_rank_order": "release_ack_recv_i_if_required_then_data_send_i_then_fill_ack_recv_i",
                "tile_branching": "tile1_head_main_chain_tile2_immediate_side_branch_tile3_plus_release_gated",
            },
            "prediction_profile": {
                "prefill_consume": self.prefill_consume,
                "decode_consume": {
                    "kind": "profiled_consume",
                    "profile_id": "decode-service-profile-v1",
                    "compute_ns_total": compute_total,
                    "hbm_bytes_total": hbm_total,
                    "resource_key": self._resource_key,
                    "share_rule": "existing_action_work_graph_byte_share_vector",
                },
            },
            "execution_profile": {
                "data_legs": "ordinary_comm_send_recv_path",
                "endpoint_hbm_rule": "one_original_COMM_READ_or_COMM_WRITE_job_per_charged_endpoint",
                "prefill_consume": "existing_body_or_marker_only",
                "decode_consume": "existing_body_nodes_only",
            },
            "logical_id_ranges": [],
            "tag_allocation_runs": [],
            "endpoint_runs": [],
            "endpoint_dependency_runs": [],
            "consume_completion_runs": [],
            "projection": {"version": 1, "chains": self.chains},
            # Prediction-only descriptors model the read/consume chains in
            # the action graph. They have no NodeStore endpoint slots and no
            # cursor anchors; ordinary service folds their virtual terminal
            # IDs directly through WorkItem.service_leg_ids.
            "anchors": [],
        }
        return PipelineDescriptor.from_wire(wire)
