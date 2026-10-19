"""General SPMD/MPMD autotune search helpers."""

from __future__ import annotations

import os
from functools import reduce
from operator import mul
from typing import Any

from atlasim import Chip

from frontend.util import set_pdeathsig

from ..io import redirect_process_output, run_pool_starmap_interruptible, safe_dump_plain
from ...ir.nodes import CoreArrayContext, KernelRegion, OpRegion
from .expressions import collect_ceildiv_operands, evaluate_static_int, replace_ceildiv_with_static_extents
from .general import append_general_operator_entry, build_data_placement_yaml_payload, build_general_task_description


_GENERAL_AUTOTUNE_WORKER_CONTEXT: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Public entrypoint and worker scheduling
# ---------------------------------------------------------------------------
def resolve_general_task_description(
    op_region: OpRegion,
    *,
    core_array_context: CoreArrayContext,
    system_config,
    placement_maps: list[dict[str, dict[str, Any]]],
    data_placement_list: list[list[dict[str, Any]]],
    element_size: int,
    intermediate_result_dir: str,
    num_workers: int | None,
) -> tuple[str, dict[str, Any]]:
    kernel_search_spaces = [
        _build_kernel_search_space(kernel_region, core_array_context=core_array_context)
        for kernel_region in op_region.kernel_regions
    ]
    if not any(search_space["autotune_enabled"] for search_space in kernel_search_spaces):
        return build_general_task_description(
            op_region,
            core_array_context=core_array_context,
            system_config=system_config,
            placement_maps=placement_maps,
            element_size=element_size,
        )

    candidate_count = reduce(mul, (space["option_count"] for space in kernel_search_spaces), 1)
    if candidate_count <= 0:
        raise ValueError(f"General autotune for '{op_region.name}' did not produce any candidate.")

    candidate_output_dir = os.path.join(intermediate_result_dir, "general_autotune", op_region.name)
    os.makedirs(candidate_output_dir, exist_ok=True)
    data_placement_config_path = os.path.join(candidate_output_dir, "data_placement.yaml")
    with open(data_placement_config_path, "w") as output_file:
        safe_dump_plain(build_data_placement_yaml_payload(data_placement_list), output_file, sort_keys=False)

    if num_workers is None or int(num_workers) <= 0:
        requested_workers = os.cpu_count() or 1
    else:
        requested_workers = int(num_workers)
    worker_count = max(min(requested_workers, candidate_count), 1)
    work_items = [
        (
            worker_id,
            worker_count,
            candidate_count,
            candidate_output_dir,
        )
        for worker_id in range(worker_count)
    ]
    print(
        f"General autotune '{op_region.name}': exploring {candidate_count} candidates "
        f"with {worker_count} workers."
    )

    global _GENERAL_AUTOTUNE_WORKER_CONTEXT
    _GENERAL_AUTOTUNE_WORKER_CONTEXT = {
        "op_region": op_region,
        "core_array_context": core_array_context,
        "system_config": system_config,
        "placement_maps": placement_maps,
        "element_size": element_size,
        "chip_config_path": system_config.chip_config_path,
        "data_placement_config_path": data_placement_config_path,
        "kernel_search_spaces": kernel_search_spaces,
    }
    try:
        if worker_count == 1:
            candidate_results = [_simulate_general_candidate_worker(*item) for item in work_items]
        else:
            candidate_results = run_pool_starmap_interruptible(
                _simulate_general_candidate_worker,
                work_items,
                processes=worker_count,
                initializer=set_pdeathsig,
                start_method="fork",
                chunksize=1,
            )
    finally:
        _GENERAL_AUTOTUNE_WORKER_CONTEXT = None

    winner = min(candidate_results, key=lambda result: (result["cycles"], tuple(-value for value in result["candidate_key"])))
    return build_general_task_description(
        op_region,
        core_array_context=core_array_context,
        system_config=system_config,
        placement_maps=placement_maps,
        element_size=element_size,
        kernel_block_values=winner["kernel_block_values"],
    )


# ---------------------------------------------------------------------------
# Search-space construction
# ---------------------------------------------------------------------------
def _build_kernel_search_space(
    kernel_region: KernelRegion,
    *,
    core_array_context: CoreArrayContext,
) -> dict[str, Any]:
    if kernel_region.autotune_enabled is not None:
        autotune_enabled = bool(kernel_region.autotune_enabled)
    else:
        autotune_attr = kernel_region.attrs.get("autotune")
        if isinstance(autotune_attr, bool):
            autotune_enabled = autotune_attr
        elif isinstance(autotune_attr, str):
            autotune_enabled = autotune_attr.lower() == "true"
        else:
            autotune_enabled = False

    tunables = _collect_kernel_tunables(kernel_region) if autotune_enabled else []
    if not tunables:
        return {
            "autotune_enabled": autotune_enabled,
            "kernel_region": kernel_region,
            "tunables": [],
            "rhs_candidate_lists": [],
            "option_count": 1,
        }

    max_candidates_per_tunable = int(core_array_context.global_kwargs.get("general_autotune_max_candidates_per_tunable", 3))
    if max_candidates_per_tunable <= 0:
        raise ValueError(
            "CoreArray kwarg 'general_autotune_max_candidates_per_tunable' must be positive "
            f"when provided, got {max_candidates_per_tunable}."
        )
    min_tile_size = int(core_array_context.global_kwargs.get("general_autotune_min_tile_size", 1))
    if min_tile_size <= 0:
        raise ValueError(
            f"CoreArray kwarg 'general_autotune_min_tile_size' must be positive when provided, got {min_tile_size}."
        )

    # Each tunable searches divisor tile sizes only. The source ceildiv rhs is
    # intentionally ignored in autotune mode; pruning is controlled by CoreArray
    # kwargs so fixed and autotuned kernels have separate semantics. When a cap
    # is needed, keep the largest tile sizes first so early worker candidates do
    # not start from tiny tiles with enormous iteration counts.
    rhs_candidate_lists: list[list[int]] = []
    for tunable in tunables:
        lhs_value = evaluate_static_int(tunable["lhs"], {}, description="General autotune ceildiv lhs")
        if lhs_value <= 0:
            raise ValueError(f"General autotune ceildiv lhs must be positive, got {lhs_value}.")

        all_divisors = [candidate_rhs for candidate_rhs in range(1, lhs_value + 1) if lhs_value % candidate_rhs == 0]
        preferred_divisors = [candidate_rhs for candidate_rhs in all_divisors if candidate_rhs >= min_tile_size]
        if not preferred_divisors:
            rhs_candidates = all_divisors[-max_candidates_per_tunable:]
        else:
            rhs_candidates = preferred_divisors[-max_candidates_per_tunable:]

        if not rhs_candidates:
            raise ValueError(f"General autotune ceildiv lhs {lhs_value} did not produce any rhs candidate.")
        rhs_candidate_lists.append(sorted(rhs_candidates, reverse=True))

    return {
        "autotune_enabled": autotune_enabled,
        "kernel_region": kernel_region,
        "tunables": tunables,
        "rhs_candidate_lists": rhs_candidate_lists,
        "option_count": reduce(mul, (len(candidates) for candidates in rhs_candidate_lists), 1),
    }


def _collect_kernel_tunables(kernel_region: KernelRegion) -> list[dict[str, Any]]:
    direct_ceildiv_by_block = {
        int(entry["block_index"]): entry
        for entry in kernel_region.attrs.get("block_ceildivs", [])
    }
    tunables: list[dict[str, Any]] = []
    for block_index, block_expr in enumerate(kernel_region.block_expressions):
        direct_ceildiv = direct_ceildiv_by_block.get(block_index)
        if direct_ceildiv is not None:
            tunables.append(
                {
                    "block_index": block_index,
                    "kind": "direct",
                    "lhs": direct_ceildiv["lhs"],
                }
            )
        else:
            for occurrence_index, (lhs_expr, _) in enumerate(collect_ceildiv_operands(block_expr)):
                tunables.append(
                    {
                        "block_index": block_index,
                        "kind": "structural",
                        "occurrence_index": occurrence_index,
                        "lhs": lhs_expr,
                    }
                )
    return tunables


# ---------------------------------------------------------------------------
# Candidate decoding and block-value materialization
# ---------------------------------------------------------------------------
def _decode_general_candidate(
    candidate_id: int,
    kernel_search_spaces: list[dict[str, Any]],
) -> tuple[tuple[int, ...], dict[int, tuple[int | None, ...]]]:
    # Decode the global candidate id lazily so the parent never materializes the
    # full Cartesian product of all kernel/tunable choices.
    kernel_option_counts = [int(space["option_count"]) for space in kernel_search_spaces]
    kernel_option_indexes = _decode_mixed_radix_index(candidate_id, kernel_option_counts)
    candidate_key_parts: list[int] = []
    kernel_block_values: dict[int, tuple[int | None, ...]] = {}
    for kernel_index, (kernel_option_index, search_space) in enumerate(zip(kernel_option_indexes, kernel_search_spaces)):
        tunables = search_space["tunables"]
        if not tunables:
            continue

        rhs_candidate_lists = search_space["rhs_candidate_lists"]
        rhs_indexes = _decode_mixed_radix_index(
            int(kernel_option_index),
            [len(candidates) for candidates in rhs_candidate_lists],
        )
        rhs_values = tuple(
            int(rhs_candidate_lists[tunable_index][rhs_index])
            for tunable_index, rhs_index in enumerate(rhs_indexes)
        )
        candidate_key_parts.extend(rhs_values)
        kernel_block_values[kernel_index] = _evaluate_kernel_block_values(
            search_space["kernel_region"],
            list(zip(tunables, rhs_values)),
        )
    return tuple(candidate_key_parts), kernel_block_values


def _decode_mixed_radix_index(index: int, radices: list[int]) -> list[int]:
    if index < 0:
        raise ValueError(f"Mixed-radix index must be non-negative, got {index}.")
    values = [0] * len(radices)
    remaining = int(index)
    for position in range(len(radices) - 1, -1, -1):
        radix = int(radices[position])
        if radix <= 0:
            raise ValueError(f"Mixed-radix radix must be positive, got {radix}.")
        values[position] = remaining % radix
        remaining //= radix
    if remaining:
        raise ValueError(f"Mixed-radix index {index} is out of range for radices {radices}.")
    return values


def _evaluate_kernel_block_values(
    kernel_region: KernelRegion,
    selected_rhs_values: list[tuple[dict[str, Any], int]],
) -> tuple[int | None, ...]:
    direct_rhs_by_block: dict[int, tuple[dict[str, Any], int]] = {}
    structural_rhs_by_block: dict[int, list[int]] = {}
    for tunable, rhs_value in selected_rhs_values:
        block_index = int(tunable["block_index"])
        if tunable["kind"] == "direct":
            direct_rhs_by_block[block_index] = (tunable, int(rhs_value))
        elif tunable["kind"] == "structural":
            structural_rhs_by_block.setdefault(block_index, []).append(int(rhs_value))
        else:
            raise ValueError(f"Unsupported general autotune tunable kind {tunable['kind']!r}.")

    values: list[int | None] = []
    for block_index, block_expr in enumerate(kernel_region.block_expressions):
        if block_index in direct_rhs_by_block:
            tunable, rhs_value = direct_rhs_by_block[block_index]
            lhs_value = evaluate_static_int(tunable["lhs"], {}, description="General autotune ceildiv lhs")
            if rhs_value <= 0 or lhs_value % rhs_value != 0:
                raise ValueError(f"General autotune rhs {rhs_value} must divide lhs {lhs_value}.")
            value = lhs_value // rhs_value
        elif block_index in structural_rhs_by_block:
            replaced_expr = replace_ceildiv_with_static_extents(block_expr, structural_rhs_by_block[block_index], {})
            value = evaluate_static_int(replaced_expr, {}, description=f"General autotune block value {block_index}")
        else:
            value = None
        if value is not None and value <= 0:
            raise ValueError(f"General kernel block value {block_index} must be positive, got {value}.")
        values.append(value)
    return tuple(values)


# ---------------------------------------------------------------------------
# Worker execution
# ---------------------------------------------------------------------------
def _simulate_general_candidate_worker(
    worker_id: int,
    worker_count: int,
    candidate_count: int,
    candidate_output_dir: str,
) -> dict[str, Any]:
    if _GENERAL_AUTOTUNE_WORKER_CONTEXT is None:
        raise RuntimeError("General autotune worker context is not initialized.")
    worker_dir = os.path.join(candidate_output_dir, f"worker_{worker_id}")
    os.makedirs(worker_dir, exist_ok=True)
    operator_description_config_path = os.path.join(worker_dir, "operator_description.yaml")
    progress_path = os.path.join(worker_dir, "worker_progress.log")
    simulator_log_path = os.path.join(worker_dir, "worker_output.log")
    worker_candidate_count = 0 if worker_id >= candidate_count else (candidate_count - 1 - worker_id) // worker_count + 1
    progress_interval = max(worker_candidate_count // 20, 1)
    best_result: dict[str, Any] | None = None

    with open(progress_path, "w") as progress_file:
        progress_file.write(
            f"worker_id={worker_id} worker_count={worker_count} candidate_count={worker_candidate_count}\n"
        )
        progress_file.flush()
        # Match GEMM autotune: each worker owns a deterministic stride through
        # the candidate id space and only returns its local best result.
        for local_index, candidate_id in enumerate(range(worker_id, candidate_count, worker_count), start=1):
            candidate_key, kernel_block_values = _decode_general_candidate(
                candidate_id,
                _GENERAL_AUTOTUNE_WORKER_CONTEXT["kernel_search_spaces"],
            )
            task_payload = build_general_task_description(
                _GENERAL_AUTOTUNE_WORKER_CONTEXT["op_region"],
                core_array_context=_GENERAL_AUTOTUNE_WORKER_CONTEXT["core_array_context"],
                system_config=_GENERAL_AUTOTUNE_WORKER_CONTEXT["system_config"],
                placement_maps=_GENERAL_AUTOTUNE_WORKER_CONTEXT["placement_maps"],
                element_size=_GENERAL_AUTOTUNE_WORKER_CONTEXT["element_size"],
                kernel_block_values=kernel_block_values,
            )[1]

            operator_entries: list[dict[str, Any]] = []
            append_general_operator_entry(
                task_payload=task_payload,
                operator_entries=operator_entries,
                intermediate_result_dir=worker_dir,
            )
            with open(operator_description_config_path, "w") as output_file:
                safe_dump_plain({"operator": operator_entries}, output_file, sort_keys=False)

            with redirect_process_output(simulator_log_path):
                chip = Chip(
                    _GENERAL_AUTOTUNE_WORKER_CONTEXT["chip_config_path"],
                    operator_description_config_path,
                    _GENERAL_AUTOTUNE_WORKER_CONTEXT["data_placement_config_path"],
                )
                performance = chip.simulate()
                del chip

            result = {
                "candidate_id": candidate_id,
                "candidate_key": tuple(candidate_key),
                "kernel_block_values": kernel_block_values,
                "cycles": int(performance.e2e_stats.e2e_cycles),
            }
            if best_result is None or (
                result["cycles"],
                tuple(-value for value in result["candidate_key"]),
            ) < (
                best_result["cycles"],
                tuple(-value for value in best_result["candidate_key"]),
            ):
                best_result = result

            if local_index == 1 or local_index == worker_candidate_count or local_index % progress_interval == 0:
                progress_file.write(
                    f"done={local_index}/{worker_candidate_count} candidate_id={candidate_id} "
                    f"candidate_key={result['candidate_key']} "
                    f"cycles={result['cycles']} best_cycles={best_result['cycles']}\n"
                )
                progress_file.flush()
            del task_payload, operator_entries, performance

    if best_result is None:
        raise ValueError(f"General autotune worker {worker_id} did not receive any candidate.")
    with open(os.path.join(worker_dir, "worker_best.yaml"), "w") as output_file:
        safe_dump_plain(best_result, output_file, sort_keys=False)
    return best_result


__all__ = ["resolve_general_task_description"]
