"""Cloud GEMM task-description generation driven by AST loop order and autotune semantics."""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from itertools import product
from typing import Any

import yaml
from atlasim import Chip, SimulatorOperatorType

from frontend.util import _KB, get_factors, make_dram_task, partition_list, set_pdeathsig

from ...ir.expr import collect_symbols, is_expr
from ...ir.nodes import KernelRegion, OpAction, OpRegion
from ..io import redirect_process_output, run_pool_starmap_interruptible, safe_dump_plain
from ..common.scalars import to_python_int

_GEMM_AXIS_ORDER = ("M", "K", "N")
_ROLE_AXES = {
    "input": ("M", "K"),
    "weight": ("K", "N"),
    "output": ("M", "N"),
}


@dataclass(slots=True)
class CloudGemmScheduleSignature:
    autotune_enabled: bool
    axis_order: tuple[str, ...]
    source_tile_shape: tuple[int, int, int]


# ---------------------------------------------------------------------------
# Shared shape / placement helpers
# ---------------------------------------------------------------------------

def _placement_matrix_shape(placement: dict[str, Any]) -> tuple[int, int]:
    shape = placement["shape"]
    if len(shape) != 2:
        raise ValueError(f"Cloud GEMM placement '{placement['name']}' must stay 2D, got {shape}.")
    return int(shape[0]), int(shape[1])


# ---------------------------------------------------------------------------
# AST schedule inspection helpers
# ---------------------------------------------------------------------------

def _iter_actions_with_paths(kernel_region: KernelRegion):
    # Depth-first traversal over every captured action, annotated with the loop
    # path that surrounds it. Later stages recover loop order from this path.
    def walk_loop(loop_region, loop_path):
        current_path = loop_path + (loop_region,)
        for action in loop_region.actions:
            yield current_path, action
        for child_loop in loop_region.child_loops:
            yield from walk_loop(child_loop, current_path)

    for action in kernel_region.actions:
        yield tuple(), action
    for child_loop in kernel_region.child_loops:
        yield from walk_loop(child_loop, tuple())


def _collect_var_names(expr: Any) -> set[str]:
    # We only need variable identities here; the exact affine form is handled
    # later when grouped tile sizes are chosen.
    if not is_expr(expr):
        return set()
    return collect_symbols(expr)


def _parse_autotune_enabled(kernel_region: KernelRegion) -> bool:
    if kernel_region.autotune_enabled is not None:
        return bool(kernel_region.autotune_enabled)
    autotune_attr = kernel_region.attrs.get("autotune")
    if isinstance(autotune_attr, bool):
        return autotune_attr
    if isinstance(autotune_attr, str):
        return autotune_attr.lower() == "true"
    return False


def _infer_axis_mapping(
    *,
    kernel_region: KernelRegion,
    input_name: str,
    output_name: str,
    weight_prefix: str,
) -> dict[str, str]:
    axis_by_var: dict[str, str] = {}
    for _loop_path, action in _iter_actions_with_paths(kernel_region):
        access = None
        role = None
        if action.action_kind in ("matrix_compute", "vector_compute"):
            pass
        elif action.action_kind == "tensor_copy":
            global_input = next((access for access in action.input_accesses if _is_tensor_access(access)), None)
            global_output = next((access for access in action.output_accesses if _is_tensor_access(access)), None)
            if global_input is not None:
                if global_input.tensor_name == input_name:
                    role = "input"
                elif global_input.tensor_name.startswith(weight_prefix):
                    role = "weight"
                else:
                    access = None
                    role = None
                    continue
                access = global_input
            elif global_output is not None:
                if global_output.tensor_name != output_name:
                    access = None
                    role = None
                else:
                    role = "output"
                    access = global_output
        else:
            raise ValueError(f"Unsupported cloud GEMM action kind {action.action_kind!r}.")
        if access is None or role is None:
            continue
        if len(access.index_expressions) != 2:
            raise ValueError(f"Cloud GEMM access for '{access.tensor_name}' must stay rank-2, got {access.index_expressions}.")
        for dim, axis_name in enumerate(_ROLE_AXES[role]):
            for var_name in _collect_var_names(access.index_expressions[dim]):
                previous = axis_by_var.get(var_name)
                if previous is not None and previous != axis_name:
                    raise ValueError(
                        f"Loop variable '{var_name}' is inconsistently mapped to both '{previous}' and '{axis_name}'."
                    )
                axis_by_var[var_name] = axis_name
    return axis_by_var


def extract_cloud_gemm_schedule_signature(
    *,
    op_region: OpRegion,
    input_name: str,
    output_name: str,
    weight_prefix: str,
) -> CloudGemmScheduleSignature:
    if len(op_region.kernel_regions) != 1:
        raise ValueError(f"Cloud GEMM '{op_region.name}' expects exactly one kernel region, got {len(op_region.kernel_regions)}.")
    kernel_region = op_region.kernel_regions[0]
    autotune_enabled = _parse_autotune_enabled(kernel_region)
    axis_by_var = _infer_axis_mapping(
        kernel_region=kernel_region,
        input_name=input_name,
        output_name=output_name,
        weight_prefix=weight_prefix,
    )

    matrix_path: tuple[Any, ...] | None = None
    matrix_action: OpAction | None = None
    for loop_path, action in _iter_actions_with_paths(kernel_region):
        if action.action_kind == "matrix_compute":
            matrix_path = loop_path
            matrix_action = action
            break
    if matrix_path is None or matrix_action is None:
        raise ValueError(f"Cloud GEMM '{op_region.name}' is missing matrix_compute action.")

    # Axis inference tells us what each loop variable means logically; walking
    # the matrix_compute loop path recovers the actual serial execution order.
    axis_order: list[str] = []
    for loop_region in matrix_path:
        if loop_region.loop_kind == "serial":
            loop_var = loop_region.loop_variable
            if loop_var is None:
                continue
            axis_name = axis_by_var.get(loop_var)
            if axis_name is None:
                continue
            if axis_name not in axis_order:
                axis_order.append(axis_name)
        elif loop_region.loop_kind in ("python-range", "general"):
            pass
        else:
            raise ValueError(f"Unsupported cloud GEMM loop kind {loop_region.loop_kind!r}.")
    if tuple(axis_order) != ("M", "N", "K"):
        # Keep non-default orders legal, but require a full M/K/N mapping.
        missing = [axis for axis in _GEMM_AXIS_ORDER if axis not in axis_order]
        if missing:
            raise ValueError(
                f"Cloud GEMM '{op_region.name}' could not recover a full axis order from AST. "
                f"Recovered {axis_order}, missing {missing}."
            )

    source_tile_shape_values: list[int] = []
    for axis_name in ("M", "K", "N"):
        source_tile_shape_values.append(
            to_python_int(matrix_action.attrs[axis_name], description=f"matrix_compute axis '{axis_name}'")
        )
    return CloudGemmScheduleSignature(
        autotune_enabled=autotune_enabled,
        axis_order=tuple(axis_order),
        source_tile_shape=tuple(source_tile_shape_values),
    )


def _is_tensor_access(access: Any) -> bool:
    return access.attrs.get("memory_space") == "tensor"


# ---------------------------------------------------------------------------
# Task construction helpers
# ---------------------------------------------------------------------------

def _matrix_and_vector_description(*, element_size: int, tM: int, tK: int, tN: int) -> dict[str, Any]:
    mac_count = tM * tN * tK
    vec_count = tM * tN
    return {
        "matrix": [{"name": "gemm_tile", "mac_count": mac_count}],
        "vector": [{"name": "gemm_tile_accumulation", "vec_count": vec_count}],
        "buffer_load": [
            {"name": "gemm_tile_load", "byte_count": element_size * (tM * tK + tK * tN), "is_write": False},
            {"name": "gemm_tile_accumulation_load", "byte_count": element_size * (tM * tN), "is_write": False},
        ],
        "buffer_store": [
            {"name": "gemm_tile_store", "byte_count": element_size * (tM * tN), "is_write": True},
            {"name": "gemm_tile_accumulation_store", "byte_count": element_size * (tM * tN), "is_write": True},
        ],
    }


def _access_window(
    *,
    axes: tuple[str, str],
    coords: dict[str, int],
    tile_shape: dict[str, int],
    dim_shape: dict[str, int],
) -> tuple[list[int], list[int]]:
    base = [coords[axis] * tile_shape[axis] for axis in axes]
    extent = [min(tile_shape[axis], dim_shape[axis] - base_dim) for axis, base_dim in zip(axes, base)]
    return base, extent


def _enumerate_compute_coords(axis_order: tuple[str, ...], axis_counts: dict[str, int]) -> list[dict[str, int]]:
    coords: list[dict[str, int]] = []

    def visit(depth: int, current: dict[str, int]) -> None:
        if depth == len(axis_order):
            full = {axis: 0 for axis in _GEMM_AXIS_ORDER}
            full.update(current)
            coords.append(full)
            return
        axis = axis_order[depth]
        for index_value in range(axis_counts[axis]):
            current[axis] = index_value
            visit(depth + 1, current)
        current.pop(axis, None)

    visit(0, {})
    return coords


def _single_shot_dram_task(
    *,
    name: str,
    is_write: bool,
    access_base: list[int],
    access_extent: list[int],
    init_iter: int,
) -> dict[str, Any]:
    rank = len(access_base)
    return make_dram_task(
        name=name,
        is_write=is_write,
        access_base=access_base,
        access_extent=access_extent,
        access_stride_add=[1] * rank,
        access_offset_add=[0] * rank,
        init_iter=init_iter,
        stride_iter=1,
        total_iter=1,
    )


def _build_cloud_dram_tasks(
    *,
    input_name: str,
    output_name: str,
    weight_name: str,
    M: int,
    K: int,
    N: int,
    tM: int,
    tK: int,
    tN: int,
    axis_order: tuple[str, ...],
) -> list[dict[str, Any]]:
    # Grouped compute tiles still come from fixed tiles or autotune/DSE, but
    # DRAM access order follows the AST-derived serial loop order exactly.
    axis_counts = {
        "M": math.ceil(M / tM),
        "K": math.ceil(K / tK),
        "N": math.ceil(N / tN),
    }
    tile_shape = {"M": int(tM), "K": int(tK), "N": int(tN)}
    dim_shape = {"M": M, "K": K, "N": N}
    compute_coords = _enumerate_compute_coords(axis_order, axis_counts)

    dram_tasks: list[dict[str, Any]] = []
    current_input_key: tuple[int, int] | None = None
    current_weight_key: tuple[int, int] | None = None
    current_output_key: tuple[int, int] | None = None
    current_output_coords: dict[str, int] | None = None
    current_output_last_compute = -1
    seen_output_keys: set[tuple[int, int]] = set()

    for compute_index, coords in enumerate(compute_coords):
        input_key = (coords["M"], coords["K"])
        if input_key != current_input_key:
            access_base, access_extent = _access_window(
                axes=_ROLE_AXES["input"],
                coords=coords,
                tile_shape=tile_shape,
                dim_shape=dim_shape,
            )
            dram_tasks.append(
                _single_shot_dram_task(
                    name=input_name,
                    is_write=False,
                    access_base=access_base,
                    access_extent=access_extent,
                    init_iter=compute_index,
                )
            )
            current_input_key = input_key

        weight_key = (coords["K"], coords["N"])
        if weight_key != current_weight_key:
            access_base, access_extent = _access_window(
                axes=_ROLE_AXES["weight"],
                coords=coords,
                tile_shape=tile_shape,
                dim_shape=dim_shape,
            )
            dram_tasks.append(
                _single_shot_dram_task(
                    name=weight_name,
                    is_write=False,
                    access_base=access_base,
                    access_extent=access_extent,
                    init_iter=compute_index,
                )
            )
            current_weight_key = weight_key

        output_key = (coords["M"], coords["N"])
        if current_output_key is None:
            current_output_key = output_key
            current_output_coords = dict(coords)
            current_output_last_compute = compute_index
            seen_output_keys.add(output_key)
        elif output_key != current_output_key:
            assert current_output_coords is not None
            access_base, access_extent = _access_window(
                axes=_ROLE_AXES["output"],
                coords=current_output_coords,
                tile_shape=tile_shape,
                dim_shape=dim_shape,
            )
            dram_tasks.append(
                _single_shot_dram_task(
                    name=output_name,
                    is_write=True,
                    access_base=access_base,
                    access_extent=access_extent,
                    init_iter=current_output_last_compute + 2,
                )
            )
            if output_key in seen_output_keys:
                access_base, access_extent = _access_window(
                    axes=_ROLE_AXES["output"],
                    coords=coords,
                    tile_shape=tile_shape,
                    dim_shape=dim_shape,
                )
                dram_tasks.append(
                    _single_shot_dram_task(
                        name=output_name,
                        is_write=False,
                        access_base=access_base,
                        access_extent=access_extent,
                        init_iter=compute_index,
                    )
                )
            current_output_key = output_key
            current_output_coords = dict(coords)
            current_output_last_compute = compute_index
            seen_output_keys.add(output_key)
        else:
            current_output_last_compute = compute_index

    if current_output_coords is not None:
        access_base, access_extent = _access_window(
            axes=_ROLE_AXES["output"],
            coords=current_output_coords,
            tile_shape=tile_shape,
            dim_shape=dim_shape,
        )
        dram_tasks.append(
            _single_shot_dram_task(
                name=output_name,
                is_write=True,
                access_base=access_base,
                access_extent=access_extent,
                init_iter=current_output_last_compute + 2,
            )
        )

    dram_tasks.sort(key=lambda task: (task["init_iter"], int(task["is_write"]), task["name"], task["access_base"]))
    return dram_tasks


def _build_cloud_comp_description(
    *,
    op_name: str,
    input_name: str,
    output_name: str,
    weight_name: str,
    element_size: int,
    M: int,
    K: int,
    N: int,
    tM: int,
    tK: int,
    tN: int,
    axis_order: tuple[str, ...],
) -> dict[str, Any]:
    execution = _matrix_and_vector_description(element_size=element_size, tM=tM, tK=tK, tN=tN)
    execution["dram"] = _build_cloud_dram_tasks(
        input_name=input_name,
        output_name=output_name,
        weight_name=weight_name,
        M=M,
        K=K,
        N=N,
        tM=tM,
        tK=tK,
        tN=tN,
        axis_order=axis_order,
    )
    return {
        "operator": [
            {
                "name": op_name,
                "type": str(SimulatorOperatorType.GEMM),
                "iteration": math.ceil(M / tM) * math.ceil(K / tK) * math.ceil(N / tN),
                "execution": execution,
            }
        ]
    }


# ---------------------------------------------------------------------------
# Autotune / cache resolution
# ---------------------------------------------------------------------------

def _explore_cloud_gemm_worker(
    input_placement: dict[str, Any],
    output_placement: dict[str, Any],
    weight_placement_list: list[dict[str, Any]],
    op_name: str,
    batch_count: int,
    element_size: int,
    chip_config_path: str,
    tiling_factor_combination_list: list[tuple[int, int, int]],
    schedule_signature: CloudGemmScheduleSignature,
    worker_id: int,
    cur_op_dir: str,
) -> dict[str, Any]:
    cur_worker_dir = os.path.join(cur_op_dir, f"worker_{worker_id}")
    os.makedirs(cur_worker_dir, exist_ok=True)

    data_placement_config = {"tensor": [input_placement, output_placement, *weight_placement_list]}
    data_placement_config_path = os.path.join(cur_worker_dir, "data_placement.yaml")
    with open(data_placement_config_path, "w") as f:
        safe_dump_plain(data_placement_config, f, sort_keys=False)

    input_name = input_placement["name"]
    output_name = output_placement["name"]
    M, K = _placement_matrix_shape(input_placement)
    _, N = _placement_matrix_shape(output_placement)

    best_latency = float("inf")
    best_result: tuple[dict[str, Any], tuple[int, int, int]] | None = None
    comp_description_path = os.path.join(cur_worker_dir, "comp_description.yaml")
    for tM, tK, tN in tiling_factor_combination_list:
        best_weight_name = weight_placement_list[0]["name"]
        comp_description = _build_cloud_comp_description(
            op_name=op_name,
            input_name=input_name,
            output_name=output_name,
            weight_name=best_weight_name,
            element_size=element_size,
            M=M,
            K=K,
            N=N,
            tM=tM,
            tK=tK,
            tN=tN,
            axis_order=schedule_signature.axis_order,
        )
        with open(comp_description_path, "w") as f:
            safe_dump_plain(comp_description, f, sort_keys=False)

        log_file_path = os.path.join(cur_worker_dir, "worker_output.log")
        with redirect_process_output(log_file_path):
            chip = Chip(chip_config_path, comp_description_path, data_placement_config_path)
            performance = chip.simulate()
            del chip

        latency = performance.e2e_stats.e2e_cycles
        if latency < best_latency:
            best_latency = latency
            best_result = (comp_description, (tM, tK, tN))

    if best_result is None:
        raise ValueError(f"Cloud GEMM worker for '{op_name}' did not find any legal tiling.")

    best_comp_description, best_tiling = best_result
    all_batch_comp_description = []
    for batch_index in range(batch_count):
        current_description = copy.deepcopy(best_comp_description["operator"][0])
        current_description["execution"]["dram"] = [
            copy.deepcopy(task) for task in current_description["execution"]["dram"]
        ]
        for dram_task in current_description["execution"]["dram"]:
            if dram_task["name"] == weight_placement_list[0]["name"]:
                dram_task["name"] = weight_placement_list[batch_index]["name"]
        all_batch_comp_description.append(current_description)
    return {
        "latency": best_latency,
        "comp_description_list": all_batch_comp_description,
        "tiling_factors": best_tiling,
    }


def resolve_cloud_gemm_task_descriptions(
    *,
    op_region: OpRegion,
    op_name: str,
    input_placement: dict[str, Any],
    output_placement: dict[str, Any],
    weight_placement_list: list[dict[str, Any]],
    element_size: int,
    chip_config,
    chip_config_path: str,
    num_workers: int | None,
    intermediate_result_dir: str,
    gemm_tiling_cache_dir: str,
    min_tile_shape: tuple[int, int, int],
) -> dict[str, Any]:
    cache_base_dir = gemm_tiling_cache_dir if gemm_tiling_cache_dir else intermediate_result_dir
    cur_op_dir = os.path.join(cache_base_dir, op_name)
    os.makedirs(cur_op_dir, exist_ok=True)

    batch_count = len(weight_placement_list)
    M, K = _placement_matrix_shape(input_placement)
    _, N = _placement_matrix_shape(output_placement)
    weight_prefix = f"weight_{op_name}"
    schedule_signature = extract_cloud_gemm_schedule_signature(
        op_region=op_region,
        input_name=input_placement["name"],
        output_name=output_placement["name"],
        weight_prefix=weight_prefix,
    )

    cache_file_path = os.path.join(cur_op_dir, "gemm_opt_general_layout.yaml")
    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as f:
            cached = yaml.safe_load(f)
        schedule_compatible = (
            cached is not None
            and "axis_order" in cached
            and "autotune_enabled" in cached
            and tuple(cached["axis_order"]) == schedule_signature.axis_order
            and bool(cached["autotune_enabled"]) == schedule_signature.autotune_enabled
        )
        if (
            cached is not None
            and cached.get("M") == M
            and cached.get("K") == K
            and cached.get("N") == N
            and cached.get("B") == batch_count
            and cached.get("element_size") == element_size
            and schedule_compatible
        ):
            return {
                "latency": float(cached["latency"]),
                "comp_description_list": cached["comp_description_list"],
                "tiling_factors": tuple(cached["tiling_factors"]),
            }

    if schedule_signature.autotune_enabled:
        min_tM, min_tK, min_tN = min_tile_shape
        tM_list = get_factors(M, min_tM)
        tK_list = get_factors(K, min_tK)
        tN_list = get_factors(N, min_tN)
        while not tM_list:
            min_tM = max(min_tM // 2, 1)
            tM_list = get_factors(M, min_tM)
        while not tK_list:
            min_tK = max(min_tK // 2, 1)
            tK_list = get_factors(K, min_tK)
        while not tN_list:
            min_tN = max(min_tN // 2, 1)
            tN_list = get_factors(N, min_tN)
        legal_tiling_factors = []
        buffer_size = chip_config.buffer_config.buffer_size * _KB
        for tM, tK, tN in product(tM_list, tK_list, tN_list):
            if element_size * (tM * tK + tK * tN + 2 * tM * tN) > buffer_size / 2:
                continue
            legal_tiling_factors.append((tM, tK, tN))
        if not legal_tiling_factors:
            raise ValueError(f"Cloud GEMM '{op_name}' has no legal tiling candidates.")
    else:
        legal_tiling_factors = [schedule_signature.source_tile_shape]

    effective_num_workers = 1 if num_workers is None else int(num_workers)
    worker_count = max(min(effective_num_workers, len(legal_tiling_factors)), 1)
    tiling_factor_partitions = partition_list(legal_tiling_factors, worker_count)
    if len(legal_tiling_factors) == 1:
        results = [
            _explore_cloud_gemm_worker(
                input_placement,
                output_placement,
                weight_placement_list,
                op_name,
                batch_count,
                element_size,
                chip_config_path,
                tiling_factor_partitions[0],
                schedule_signature,
                0,
                cur_op_dir,
            )
        ]
    else:
        results = run_pool_starmap_interruptible(
            _explore_cloud_gemm_worker,
            [
                (
                    input_placement,
                    output_placement,
                    weight_placement_list,
                    op_name,
                    batch_count,
                    element_size,
                    chip_config_path,
                    tiling_factor_partition,
                    schedule_signature,
                    worker_id,
                    cur_op_dir,
                )
                for worker_id, tiling_factor_partition in enumerate(tiling_factor_partitions)
            ],
            processes=worker_count,
            initializer=set_pdeathsig,
            start_method="fork",
        )

    best_latency = float("inf")
    best_result: dict[str, Any] | None = None
    for result in results:
        if result["latency"] < best_latency:
            best_latency = result["latency"]
            best_result = result

    if best_result is None:
        raise ValueError(f"Cloud GEMM '{op_name}' did not produce any candidate result.")

    best_comp_description_list = best_result["comp_description_list"]
    best_tiling_factors = best_result["tiling_factors"]
    cache_data = {
        "M": M,
        "K": K,
        "N": N,
        "B": batch_count,
        "element_size": element_size,
        "axis_order": list(schedule_signature.axis_order),
        "autotune_enabled": schedule_signature.autotune_enabled,
        "latency": float(best_latency),
        "comp_description_list": best_comp_description_list,
        "tiling_factors": list(best_tiling_factors),
    }
    with open(cache_file_path, "w") as f:
        safe_dump_plain(cache_data, f, sort_keys=False)
    return {
        "latency": best_latency,
        "comp_description_list": best_comp_description_list,
        "tiling_factors": best_tiling_factors,
    }


__all__ = [
    "CloudGemmScheduleSignature",
    "extract_cloud_gemm_schedule_signature",
    "resolve_cloud_gemm_task_descriptions",
]
