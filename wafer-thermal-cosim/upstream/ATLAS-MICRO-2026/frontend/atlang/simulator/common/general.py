"""General SPMD/MPMD materialization helpers for simulator YAML outputs."""

from __future__ import annotations

import copy
import math
import os
from typing import Any

from frontend.util import make_dram_task

from ...ir.dtype import dtype_byte_size
from ..io import safe_dump_plain
from ...ir.nodes import CoreArrayContext, KernelRegion, LoopRegion, OpAction, OpRegion, SimulatorMetadataSnapshot, TensorAccess, TensorDecl
from .dram import finalize_dram_tasks
from .expressions import evaluate_static_bool, evaluate_static_int, substitute_and_simplify
from .layout import align_base_addr, build_layout_record, normalize_layout_tuple, tensor_volume_bytes
from .noc import assign_general_execution_noc_flit_ids


# -----------------------------------------------------------------------------
# Public materialization entry points
# -----------------------------------------------------------------------------
def build_general_data_placement_list(
    snapshot: SimulatorMetadataSnapshot,
    *,
    base_data_placement_list: list[dict[str, Any]],
    element_size: int,
    dram_row_size: int,
    system_config,
) -> list[list[dict[str, Any]]]:
    """Build one complete tensor placement list for each simulator core."""
    core_num = int(system_config.chip_config.core_num)
    per_core_placements = [copy.deepcopy(base_data_placement_list) for _ in range(core_num)]
    placement_names_by_core = [{placement["name"] for placement in placements} for placements in per_core_placements]
    next_base_addr = _next_available_base_addr(base_data_placement_list, dram_row_size)
    generated_placements: dict[str, dict[str, Any]] = {}

    for core_array_context in snapshot.core_array_contexts:
        for op_region in core_array_context.op_regions:
            if op_region.op_category == "general":
                next_base_addr = append_general_region_data_placements(
                    op_region,
                    core_array_context=core_array_context,
                    per_core_placements=per_core_placements,
                    placement_names_by_core=placement_names_by_core,
                    generated_placements=generated_placements,
                    tensor_declarations=snapshot.tensor_declarations,
                    base_addr=next_base_addr,
                    element_size=element_size,
                    dram_row_size=dram_row_size,
                )
            elif op_region.op_category in ("gemm", "communication", "decode-attention"):
                pass
            else:
                raise ValueError(f"Unsupported op category {op_region.op_category!r} while building general placement.")

    return per_core_placements


def build_data_placement_yaml_payload(data_placement_list: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Wrap per-core placement lists in the simulator `core_tensor` YAML shape."""
    return {
        "core_tensor": [
            {
                "core_id": core_id,
                "tensor": core_data_placement_list,
            }
            for core_id, core_data_placement_list in enumerate(data_placement_list)
        ]
    }


def build_general_task_descriptions(
    snapshot: SimulatorMetadataSnapshot,
    *,
    system_config,
    data_placement_list: list[list[dict[str, Any]]],
    element_size: int,
) -> list[tuple[str, dict[str, Any]]]:
    """Materialize captured general regions into simulator task-description payloads."""
    core_num = int(system_config.chip_config.core_num)
    if len(data_placement_list) != core_num:
        raise ValueError(
            f"General data placement must provide one complete tensor list for each core; "
            f"expected {core_num}, got {len(data_placement_list)}."
        )
    placement_maps = [{placement["name"]: placement for placement in placements} for placements in data_placement_list]
    task_descriptions: list[tuple[str, dict[str, Any]]] = []
    for core_array_context in snapshot.core_array_contexts:
        for op_region in core_array_context.op_regions:
            if op_region.op_category == "general":
                task_descriptions.append(
                    build_general_task_description(
                        op_region,
                        core_array_context=core_array_context,
                        system_config=system_config,
                        placement_maps=placement_maps,
                        element_size=element_size,
                    )
                )
            elif op_region.op_category in ("gemm", "communication", "decode-attention"):
                pass
            else:
                raise ValueError(f"Unsupported op category {op_region.op_category!r} while building general tasks.")
    return task_descriptions


def append_general_region_data_placements(
    op_region: OpRegion,
    *,
    core_array_context: CoreArrayContext,
    per_core_placements: list[list[dict[str, Any]]],
    placement_names_by_core: list[set[str]],
    generated_placements: dict[str, dict[str, Any]],
    tensor_declarations: dict[str, TensorDecl],
    base_addr: int,
    element_size: int,
    dram_row_size: int,
) -> int:
    core_num = len(per_core_placements)
    if core_array_context.core_num != core_num:
        raise ValueError(
            f"General op '{op_region.name}' is captured for {core_array_context.core_num} cores, "
            f"but system config has {core_num} cores."
        )
    if op_region.region_kind == "spmd":
        tensor_names = _collect_global_tensor_names(op_region.kernel_regions[0])
        for core_id in range(core_num):
            base_addr = _append_missing_general_placements(
                tensor_names,
                core_id=core_id,
                per_core_placements=per_core_placements,
                placement_names_by_core=placement_names_by_core,
                generated_placements=generated_placements,
                tensor_declarations=tensor_declarations,
                base_addr=base_addr,
                element_size=element_size,
                dram_row_size=dram_row_size,
            )
        return base_addr

    for kernel_region in op_region.kernel_regions:
        tensor_names = _collect_global_tensor_names(kernel_region)
        for core_id in kernel_region.core_list or []:
            base_addr = _append_missing_general_placements(
                tensor_names,
                core_id=int(core_id),
                per_core_placements=per_core_placements,
                placement_names_by_core=placement_names_by_core,
                generated_placements=generated_placements,
                tensor_declarations=tensor_declarations,
                base_addr=base_addr,
                element_size=element_size,
                dram_row_size=dram_row_size,
            )
    return base_addr


def build_general_task_description(
    op_region: OpRegion,
    *,
    core_array_context: CoreArrayContext,
    system_config,
    placement_maps: list[dict[str, dict[str, Any]]],
    element_size: int,
    kernel_block_values: dict[int, tuple[int | None, ...]] | None = None,
) -> tuple[str, dict[str, Any]]:
    core_num = int(system_config.chip_config.core_num)
    if core_array_context.core_num != core_num:
        raise ValueError(
            f"General op '{op_region.name}' is captured for {core_array_context.core_num} cores, "
            f"but system config has {core_num} cores."
        )
    if op_region.region_kind == "spmd":
        spmd_kernel_index = 0
        spmd_kernel_region = op_region.kernel_regions[spmd_kernel_index]
        spmd_block_values = None
        if kernel_block_values is not None:
            spmd_block_values = kernel_block_values.get(spmd_kernel_index)
        # SPMD general shares one execution description across all cores. The
        # per-core placement lists are complete and identical for this shared
        # materialization, so core 0 is the representative placement map used
        # to resolve tensor shapes and byte counts.
        spmd_representative_core_id = 0
        spmd_placement_map = placement_maps[spmd_representative_core_id] if placement_maps else {}
        execution = _materialize_kernel_execution(
            spmd_kernel_region,
            env={},
            placement_map=spmd_placement_map,
            default_element_size=element_size,
            system_config=system_config,
            block_values=spmd_block_values,
        )
        assign_general_execution_noc_flit_ids([execution])
        return (
            "general",
            {
                "operator": {
                    "name": op_region.name,
                    "type": "spmd-general",
                    "iteration": len(execution),
                },
                "execution": execution,
            },
        )

    core_identifier_name = op_region.attrs.get("core_identifier")
    per_core_execution = [[] for _ in range(core_num)]
    for kernel_index, kernel_region in enumerate(op_region.kernel_regions):
        block_values_for_kernel = None
        if kernel_block_values is not None:
            block_values_for_kernel = kernel_block_values.get(kernel_index)
        for core_id in kernel_region.core_list or []:
            env = {}
            if core_identifier_name is not None:
                env[str(core_identifier_name)] = int(core_id)
                env.setdefault("core_id", int(core_id))
            per_core_execution[int(core_id)] = _materialize_kernel_execution(
                kernel_region,
                env=env,
                placement_map=placement_maps[int(core_id)] if placement_maps else {},
                default_element_size=element_size,
                system_config=system_config,
                block_values=block_values_for_kernel,
            )
    iteration = max((len(execution) for execution in per_core_execution), default=0)
    for execution in per_core_execution:
        while len(execution) < iteration:
            execution.append(_empty_general_iteration())
    assign_general_execution_noc_flit_ids(per_core_execution)
    return (
        "general",
        {
            "operator": {
                "name": op_region.name,
                "type": "mpmd-general",
                "iteration": iteration,
            },
            "per_core_execution": per_core_execution,
        },
    )


def append_general_operator_entry(
    *,
    task_payload: dict[str, Any],
    operator_entries: list[dict[str, Any]],
    intermediate_result_dir: str,
) -> None:
    """Write general execution YAML files and append the top-level operator entry."""
    operator_entry = copy.deepcopy(task_payload["operator"])
    operator_name = str(operator_entry["name"])
    output_dir = os.path.abspath(os.path.join(intermediate_result_dir, "general", operator_name))
    os.makedirs(output_dir, exist_ok=True)

    if operator_entry["type"] == "spmd-general":
        execution_path = os.path.join(output_dir, "execution.yaml")
        with open(execution_path, "w") as output_file:
            safe_dump_plain(
                {
                    "iteration": int(operator_entry["iteration"]),
                    "execution": task_payload["execution"],
                },
                output_file,
                sort_keys=False,
            )
        operator_entry["file_prefix"] = execution_path
        operator_entries.append(operator_entry)
    elif operator_entry["type"] == "mpmd-general":
        for core_id, execution in enumerate(task_payload["per_core_execution"]):
            execution_path = os.path.join(output_dir, f"core_{core_id}.yaml")
            with open(execution_path, "w") as output_file:
                safe_dump_plain(
                    {
                        "iteration": int(operator_entry["iteration"]),
                        "execution": execution,
                    },
                    output_file,
                    sort_keys=False,
                )
        operator_entry["file_prefix"] = output_dir
        operator_entries.append(operator_entry)
    else:
        raise ValueError(f"Unsupported general operator type {operator_entry['type']!r}.")


# -----------------------------------------------------------------------------
# Data placement materialization
# -----------------------------------------------------------------------------
def _next_available_base_addr(data_placement_list: list[dict[str, Any]], dram_row_size: int) -> int:
    next_base_addr = 0
    for placement in data_placement_list:
        placement_end = align_base_addr(
            int(placement["base_addr"]),
            tensor_volume_bytes(placement["shape"], int(placement["element_size"])),
            dram_row_size,
        )
        next_base_addr = max(next_base_addr, placement_end)
    return next_base_addr


def _append_missing_general_placements(
    tensor_names: list[str],
    *,
    core_id: int,
    per_core_placements: list[list[dict[str, Any]]],
    placement_names_by_core: list[set[str]],
    generated_placements: dict[str, dict[str, Any]],
    tensor_declarations: dict[str, TensorDecl],
    base_addr: int,
    element_size: int,
    dram_row_size: int,
) -> int:
    for tensor_name in tensor_names:
        if tensor_name in placement_names_by_core[core_id]:
            continue
        placement = generated_placements.get(tensor_name)
        if placement is None:
            tensor_decl = tensor_declarations.get(tensor_name)
            if tensor_decl is None or not tensor_decl.has_layout:
                raise ValueError(f"General tensor '{tensor_name}' is missing a static global tensor declaration.")
            shape = normalize_layout_tuple(tensor_decl.shape, field_name="shape", tensor_name=tensor_name)
            strides = normalize_layout_tuple(tensor_decl.strides, field_name="strides", tensor_name=tensor_name)
            placement_element_size = _tensor_decl_element_size(tensor_decl, element_size)
            placement = build_layout_record(tensor_name, base_addr, shape, strides, placement_element_size)
            generated_placements[tensor_name] = placement
            base_addr = align_base_addr(base_addr, tensor_volume_bytes(shape, placement_element_size), dram_row_size)
        per_core_placements[core_id].append(copy.deepcopy(placement))
        placement_names_by_core[core_id].add(tensor_name)
    return base_addr


def _tensor_decl_element_size(tensor_decl: TensorDecl, default_element_size: int) -> int:
    if tensor_decl.dtype is None:
        return int(default_element_size)
    try:
        return dtype_byte_size(tensor_decl.dtype)
    except ValueError:
        return int(default_element_size)


def _collect_global_tensor_names(kernel_region: KernelRegion) -> list[str]:
    tensor_names: list[str] = []
    seen_names: set[str] = set()

    def visit(region: KernelRegion | LoopRegion) -> None:
        for item_kind, item_index in region.body_sequence:
            if item_kind == "action":
                action = region.actions[item_index]
                for access in action.input_accesses + action.output_accesses:
                    if not _is_global_access(access) or access.tensor_name in seen_names:
                        continue
                    tensor_names.append(access.tensor_name)
                    seen_names.add(access.tensor_name)
            elif item_kind == "loop":
                visit(region.child_loops[item_index])
            else:
                raise ValueError(f"Unsupported general body item kind {item_kind!r}.")

    visit(kernel_region)
    return tensor_names


# -----------------------------------------------------------------------------
# Kernel iteration flattening
# -----------------------------------------------------------------------------
def _empty_general_iteration() -> dict[str, Any]:
    return {
        "matrix": [],
        "vector": [],
        "buffer_load": [],
        "buffer_store": [],
        "dram": [],
        "noc_tx": [],
        "noc_rx": [],
    }


def _has_general_iteration_tasks(iteration_tasks: dict[str, Any]) -> bool:
    return any(bool(tasks) for tasks in iteration_tasks.values())


def _materialize_kernel_execution(
    kernel_region: KernelRegion,
    *,
    env: dict[str, int],
    placement_map: dict[str, dict[str, Any]],
    default_element_size: int,
    system_config,
    block_values: tuple[int | None, ...] | None = None,
) -> list[dict[str, Any]]:
    raw_iterations: list[list[tuple[OpAction, dict[str, int]]]] = []
    kernel_env = _bind_kernel_block_values(kernel_region, env, block_values)
    current_iteration: list[tuple[OpAction, dict[str, int]]] = []
    current_iteration = _flatten_region_body(kernel_region, kernel_env, current_iteration, raw_iterations)
    if current_iteration:
        raw_iterations.append(current_iteration)

    execution: list[dict[str, Any]] = []
    for raw_iteration in raw_iterations:
        iteration_tasks = _empty_general_iteration()
        for action, action_env in raw_iteration:
            _emit_action_tasks(
                action,
                iteration_tasks=iteration_tasks,
                env=action_env,
                placement_map=placement_map,
                default_element_size=default_element_size,
                system_config=system_config,
            )
        iteration_tasks["dram"] = finalize_dram_tasks(iteration_tasks["dram"])
        if _has_general_iteration_tasks(iteration_tasks):
            execution.append(iteration_tasks)
    return execution


def _bind_kernel_block_values(
    kernel_region: KernelRegion,
    env: dict[str, int],
    block_values: tuple[int | None, ...] | None,
) -> dict[str, int]:
    if block_values is None:
        values = [
            evaluate_static_int(block_expr, env, description=f"General kernel block value {block_index}")
            for block_index, block_expr in enumerate(kernel_region.block_expressions)
        ]
    else:
        expected_value_count = len(kernel_region.block_expressions)
        if len(block_values) != expected_value_count:
            raise ValueError(
                f"General kernel block value count mismatch: expected {expected_value_count}, got {len(block_values)}."
            )
        values = [
            evaluate_static_int(block_expr, env, description=f"General kernel block value {block_index}")
            if value is None else int(value)
            for block_index, (block_expr, value) in enumerate(zip(kernel_region.block_expressions, block_values))
        ]
    expected_value_count = len(kernel_region.block_expressions)
    if len(values) != expected_value_count:
        raise ValueError(
            f"General kernel block value count mismatch: expected {expected_value_count}, got {len(values)}."
        )
    for block_index, value in enumerate(values):
        if value <= 0:
            raise ValueError(f"General kernel block value {block_index} must be positive, got {value}.")
    kernel_env = dict(env)
    for symbol_name, block_value in zip(kernel_region.bound_symbols, values):
        kernel_env[str(symbol_name)] = int(block_value)
    return kernel_env


def _flatten_region_body(
    region: KernelRegion | LoopRegion,
    env: dict[str, int],
    current_iteration: list[tuple[OpAction, dict[str, int]]],
    raw_iterations: list[list[tuple[OpAction, dict[str, int]]]],
) -> list[tuple[OpAction, dict[str, int]]]:
    for item_kind, item_index in region.body_sequence:
        if item_kind == "action":
            action = region.actions[item_index]
            if all(
                evaluate_static_bool(condition, env, description=f"Condition {index} on action '{action.action_kind}'")
                for index, condition in enumerate(action.attrs.get("conditions", ()))
            ):
                current_iteration.append((action, dict(env)))
        elif item_kind == "loop":
            loop_region = region.child_loops[item_index]
            if loop_region.loop_kind == "serial":
                if current_iteration:
                    raw_iterations.append(current_iteration)
                    current_iteration = []
                raw_iterations.extend(_flatten_serial_loop(loop_region, env))
            elif loop_region.loop_kind in ("python-range", "general"):
                extent = evaluate_static_int(
                    loop_region.extent_expression,
                    env,
                    description=f"General {loop_region.loop_kind} loop extent",
                )
                for loop_index in range(extent):
                    child_env = _bind_loop_iteration(loop_region, loop_index, env)
                    current_iteration = _flatten_region_body(loop_region, child_env, current_iteration, raw_iterations)
            else:
                raise ValueError(f"Unsupported general loop kind {loop_region.loop_kind!r}.")
        else:
            raise ValueError(f"Unsupported general body item kind {item_kind!r}.")
    return current_iteration


def _flatten_serial_loop(loop_region: LoopRegion, env: dict[str, int]) -> list[list[tuple[OpAction, dict[str, int]]]]:
    extent = evaluate_static_int(loop_region.extent_expression, env, description="General serial loop extent")
    raw_iterations: list[list[tuple[OpAction, dict[str, int]]]] = []
    for loop_index in range(extent):
        child_env = _bind_loop_iteration(loop_region, loop_index, env)
        current_iteration: list[tuple[OpAction, dict[str, int]]] = []
        current_iteration = _flatten_region_body(loop_region, child_env, current_iteration, raw_iterations)
        if current_iteration:
            raw_iterations.append(current_iteration)
    return raw_iterations


def _bind_loop_iteration(loop_region: LoopRegion, loop_index: int, env: dict[str, int]) -> dict[str, int]:
    child_env = dict(env)
    start = evaluate_static_int(loop_region.attrs.get("start", 0), env, description="General loop start")
    step = evaluate_static_int(loop_region.attrs.get("step", 1), env, description="General loop step")
    loop_value = start + loop_index * step
    if loop_region.loop_variable is not None:
        child_env[loop_region.loop_variable] = loop_value
    iter_var = loop_region.attrs.get("iter_var")
    if isinstance(iter_var, str):
        child_env.setdefault(iter_var, loop_value)
    return child_env


# -----------------------------------------------------------------------------
# Action lowering
# -----------------------------------------------------------------------------
def _emit_action_tasks(
    action: OpAction,
    *,
    iteration_tasks: dict[str, Any],
    env: dict[str, int],
    placement_map: dict[str, dict[str, Any]],
    default_element_size: int,
    system_config,
) -> None:
    if action.action_kind == "matrix_compute":
        _emit_matrix_compute(action, iteration_tasks, env, placement_map, default_element_size)
    elif action.action_kind == "vector_compute":
        if action.attrs.get("op_name") == "clear":
            return
        _emit_vector_compute(action, iteration_tasks, env, placement_map, default_element_size)
    elif action.action_kind == "tensor_copy":
        _emit_tensor_copy(action, iteration_tasks, env, placement_map, default_element_size)
    elif action.action_kind == "noc_send":
        _emit_noc_send(action, iteration_tasks, env, placement_map, default_element_size, system_config)
    elif action.action_kind == "noc_recv":
        _emit_noc_recv(action, iteration_tasks, env, placement_map, default_element_size, system_config)
    elif action.action_kind == "dram_access":
        _emit_direct_dram_access(action, iteration_tasks)
    elif action.action_kind in ("buffer_load", "buffer_store"):
        _emit_direct_buffer_access(action, iteration_tasks, env, placement_map, default_element_size)
    else:
        raise ValueError(f"Unsupported general action kind {action.action_kind!r}.")


def _emit_matrix_compute(
    action: OpAction,
    iteration_tasks: dict[str, Any],
    env: dict[str, int],
    placement_map: dict[str, dict[str, Any]],
    default_element_size: int,
) -> None:
    _reject_global_compute_accesses(action)
    m_dim = evaluate_static_int(action.attrs["M"], env, description="matrix_compute M")
    n_dim = evaluate_static_int(action.attrs["N"], env, description="matrix_compute N")
    k_dim = evaluate_static_int(action.attrs["K"], env, description="matrix_compute K")
    mac_count = m_dim * n_dim * k_dim
    if mac_count > 0:
        iteration_tasks["matrix"].append(
            {
                "name": str(action.attrs.get("op_key") or "tl.tileop.gemm_py"),
                "mac_count": mac_count,
            }
        )
    _append_buffer_tasks_for_accesses(
        iteration_tasks["buffer_load"],
        action.input_accesses,
        is_write=False,
        name_prefix="matrix_input",
        env=env,
        placement_map=placement_map,
        default_element_size=default_element_size,
    )
    _append_buffer_tasks_for_accesses(
        iteration_tasks["buffer_store"],
        action.output_accesses,
        is_write=True,
        name_prefix="matrix_output",
        env=env,
        placement_map=placement_map,
        default_element_size=default_element_size,
    )


def _emit_vector_compute(
    action: OpAction,
    iteration_tasks: dict[str, Any],
    env: dict[str, int],
    placement_map: dict[str, dict[str, Any]],
    default_element_size: int,
) -> None:
    _reject_global_compute_accesses(action)
    op_name = str(action.attrs.get("op_name") or "vector_compute")
    _append_buffer_tasks_for_accesses(
        iteration_tasks["buffer_load"],
        action.input_accesses,
        is_write=False,
        name_prefix=f"{op_name}_input",
        env=env,
        placement_map=placement_map,
        default_element_size=default_element_size,
    )
    _append_buffer_tasks_for_accesses(
        iteration_tasks["buffer_store"],
        action.output_accesses,
        is_write=True,
        name_prefix=f"{op_name}_output",
        env=env,
        placement_map=placement_map,
        default_element_size=default_element_size,
    )
    output_vec_count = 0
    for access in action.output_accesses:
        byte_count = _access_byte_count(
            access,
            env=env,
            placement_map=placement_map,
            default_element_size=default_element_size,
        )
        element_size = _access_element_size(access, placement_map.get(access.tensor_name), default_element_size)
        if element_size > 0:
            output_vec_count = max(output_vec_count, byte_count // element_size)
    if output_vec_count > 0:
        iteration_tasks["vector"].append(
            {
                "name": op_name,
                "vec_count": output_vec_count,
            }
        )


def _emit_tensor_copy(
    action: OpAction,
    iteration_tasks: dict[str, Any],
    env: dict[str, int],
    placement_map: dict[str, dict[str, Any]],
    default_element_size: int,
) -> None:
    if len(action.input_accesses) != 1 or len(action.output_accesses) != 1:
        raise ValueError("General tensor_copy expects exactly one input access and one output access.")
    src_access = action.input_accesses[0]
    dst_access = action.output_accesses[0]
    src_global = _is_global_access(src_access)
    dst_global = _is_global_access(dst_access)

    if src_global and dst_global:
        raise ValueError("General tensor_copy does not support global-to-global copy; stage through a local buffer first.")
    if src_global:
        _append_dram_task(iteration_tasks["dram"], src_access, is_write=False, env=env, placement_map=placement_map)
        _append_buffer_task_for_access(
            iteration_tasks["buffer_store"],
            dst_access,
            is_write=True,
            name="tensor_copy_store",
            env=env,
            placement_map=placement_map,
            default_element_size=default_element_size,
        )
        return
    if dst_global:
        _append_buffer_task_for_access(
            iteration_tasks["buffer_load"],
            src_access,
            is_write=False,
            name="tensor_copy_load",
            env=env,
            placement_map=placement_map,
            default_element_size=default_element_size,
        )
        _append_dram_task(iteration_tasks["dram"], dst_access, is_write=True, env=env, placement_map=placement_map)
        return

    _append_buffer_task_for_access(
        iteration_tasks["buffer_load"],
        src_access,
        is_write=False,
        name="tensor_copy_load",
        env=env,
        placement_map=placement_map,
        default_element_size=default_element_size,
    )
    _append_buffer_task_for_access(
        iteration_tasks["buffer_store"],
        dst_access,
        is_write=True,
        name="tensor_copy_store",
        env=env,
        placement_map=placement_map,
        default_element_size=default_element_size,
    )


def _emit_noc_send(
    action: OpAction,
    iteration_tasks: dict[str, Any],
    env: dict[str, int],
    placement_map: dict[str, dict[str, Any]],
    default_element_size: int,
    system_config,
) -> None:
    if not system_config.chip_config.has_noc:
        raise ValueError("General noc_send requires an architecture with NoC.")
    if len(action.input_accesses) != 1:
        raise ValueError("General noc_send expects exactly one input access.")
    payload = action.input_accesses[0]
    if _is_global_access(payload):
        raise ValueError("General noc_send payload must be local; copy global tensors into a local buffer first.")
    byte_count = _access_byte_count(payload, env=env, placement_map=placement_map, default_element_size=default_element_size)
    if byte_count <= 0:
        return
    src_core = evaluate_static_int(action.attrs["src_core"], env, description="noc_send src_core")
    dst_core = evaluate_static_int(action.attrs["dst_core"], env, description="noc_send dst_core")
    _append_buffer_task_for_access(
        iteration_tasks["buffer_load"],
        payload,
        is_write=False,
        name=f"noc_tx{dst_core}_load",
        env=env,
        placement_map=placement_map,
        default_element_size=default_element_size,
    )
    iteration_tasks["noc_tx"].append(
        {
            "name": f"noc_tx{dst_core}",
            "is_send": True,
            "src": src_core,
            "dst": dst_core,
            "flit_num": math.ceil(byte_count / int(system_config.chip_config.noc_config.flit_size)),
            "init_flit_id": -1,
        }
    )


def _emit_noc_recv(
    action: OpAction,
    iteration_tasks: dict[str, Any],
    env: dict[str, int],
    placement_map: dict[str, dict[str, Any]],
    default_element_size: int,
    system_config,
) -> None:
    if not system_config.chip_config.has_noc:
        raise ValueError("General noc_recv requires an architecture with NoC.")
    if len(action.output_accesses) != 1:
        raise ValueError("General noc_recv expects exactly one output access.")
    payload = action.output_accesses[0]
    if _is_global_access(payload):
        raise ValueError("General noc_recv payload must be local; copy local buffers into global tensors after receive.")
    byte_count = _access_byte_count(payload, env=env, placement_map=placement_map, default_element_size=default_element_size)
    if byte_count <= 0:
        return
    src_core = evaluate_static_int(action.attrs["src_core"], env, description="noc_recv src_core")
    dst_core = evaluate_static_int(action.attrs["dst_core"], env, description="noc_recv dst_core")
    _append_buffer_task_for_access(
        iteration_tasks["buffer_store"],
        payload,
        is_write=True,
        name=f"noc_rx{src_core}_store",
        env=env,
        placement_map=placement_map,
        default_element_size=default_element_size,
    )
    iteration_tasks["noc_rx"].append(
        {
            "name": f"noc_rx{src_core}",
            "is_send": False,
            "src": src_core,
            "dst": dst_core,
            "flit_num": math.ceil(byte_count / int(system_config.chip_config.noc_config.flit_size)),
            "init_flit_id": -1,
        }
    )


def _emit_direct_dram_access(action: OpAction, iteration_tasks: dict[str, Any]) -> None:
    required_fields = (
        "name",
        "is_write",
        "access_base",
        "access_extent",
        "access_stride_add",
        "access_offset_add",
        "init_iter",
        "stride_iter",
        "total_iter",
    )
    missing_fields = [field_name for field_name in required_fields if field_name not in action.attrs]
    if missing_fields:
        raise ValueError(f"General dram_access is missing required attrs: {', '.join(missing_fields)}.")
    dram_task = {field_name: action.attrs[field_name] for field_name in required_fields}
    if int(dram_task["init_iter"]) != 0 or int(dram_task["stride_iter"]) != 1 or int(dram_task["total_iter"]) != 1:
        raise ValueError("General dram_access must set init_iter=0, stride_iter=1, total_iter=1.")
    iteration_tasks["dram"].append(dram_task)


def _emit_direct_buffer_access(
    action: OpAction,
    iteration_tasks: dict[str, Any],
    env: dict[str, int],
    placement_map: dict[str, dict[str, Any]],
    default_element_size: int,
) -> None:
    is_write = action.action_kind == "buffer_store"
    target_list = iteration_tasks["buffer_store" if is_write else "buffer_load"]
    accesses = action.output_accesses if is_write else action.input_accesses
    if not accesses and "byte_count" in action.attrs:
        byte_count = evaluate_static_int(action.attrs["byte_count"], env, description=f"{action.action_kind} byte_count")
        if byte_count > 0:
            target_list.append(
                {
                    "name": str(action.attrs.get("name") or action.action_kind),
                    "byte_count": byte_count,
                    "is_write": is_write,
                }
            )
        return
    _append_buffer_tasks_for_accesses(
        target_list,
        accesses,
        is_write=is_write,
        name_prefix=action.action_kind,
        env=env,
        placement_map=placement_map,
        default_element_size=default_element_size,
    )


def _reject_global_compute_accesses(action: OpAction) -> None:
    for access in action.input_accesses + action.output_accesses:
        if _is_global_access(access):
            raise ValueError(
                f"General {action.action_kind} cannot directly access global tensor '{access.tensor_name}'; "
                "use A.copy to stage data through a local buffer."
            )


# -----------------------------------------------------------------------------
# Tensor access and component task helpers
# -----------------------------------------------------------------------------
def _append_buffer_tasks_for_accesses(
    target_list: list[dict[str, Any]],
    accesses: list[TensorAccess],
    *,
    is_write: bool,
    name_prefix: str,
    env: dict[str, int],
    placement_map: dict[str, dict[str, Any]],
    default_element_size: int,
) -> None:
    for access_index, access in enumerate(accesses):
        _append_buffer_task_for_access(
            target_list,
            access,
            is_write=is_write,
            name=f"{name_prefix}_{access_index}",
            env=env,
            placement_map=placement_map,
            default_element_size=default_element_size,
        )


def _append_buffer_task_for_access(
    target_list: list[dict[str, Any]],
    access: TensorAccess,
    *,
    is_write: bool,
    name: str,
    env: dict[str, int],
    placement_map: dict[str, dict[str, Any]],
    default_element_size: int,
) -> None:
    if _is_global_access(access):
        return
    byte_count = _access_byte_count(access, env=env, placement_map=placement_map, default_element_size=default_element_size)
    if byte_count <= 0:
        return
    target_list.append(
        {
            "name": name,
            "byte_count": byte_count,
            "is_write": is_write,
        }
    )


def _append_dram_task(
    target_list: list[dict[str, Any]],
    access: TensorAccess,
    *,
    is_write: bool,
    env: dict[str, int],
    placement_map: dict[str, dict[str, Any]],
) -> None:
    access_base, access_extent, _ = _compute_access_window(access, env=env, placement_map=placement_map, default_element_size=1)
    if any(int(extent) <= 0 for extent in access_extent):
        return
    target_list.append(
        make_dram_task(
            name=access.tensor_name,
            is_write=is_write,
            access_base=access_base,
            access_extent=access_extent,
            access_stride_add=[1] * len(access_base),
            access_offset_add=access_extent,
            init_iter=0,
            stride_iter=1,
            total_iter=1,
        )
    )


def _compute_access_window(
    access: TensorAccess,
    *,
    env: dict[str, int],
    placement_map: dict[str, dict[str, Any]],
    default_element_size: int,
) -> tuple[list[int], list[int], int]:
    if access.slice_shape is None:
        raise ValueError(f"General access '{access.tensor_name}' is missing slice_shape.")
    placement = placement_map.get(access.tensor_name)
    access_base = [
        evaluate_static_int(index_expr, env, description=f"Access base of '{access.tensor_name}' axis {axis}")
        for axis, index_expr in enumerate(access.index_expressions)
    ]
    access_extent = []
    for axis, axis_extent in enumerate(access.slice_shape):
        axis_value = substitute_and_simplify(axis_extent, env)
        if isinstance(axis_value, bool):
            raise ValueError(f"Access extent of '{access.tensor_name}' axis {axis} must be an integer, got bool.")
        if isinstance(axis_value, int):
            access_extent.append(axis_value)
            continue
        if placement is None:
            raise ValueError(
                f"General access '{access.tensor_name}' axis {axis} still contains symbolic extent {axis_value!r} "
                "and no placement is available to resolve it."
            )
        access_extent.append(int(placement["shape"][axis]))
    element_size = _access_element_size(access, placement, default_element_size)
    return access_base, access_extent, element_size


def _access_byte_count(
    access: TensorAccess,
    *,
    env: dict[str, int],
    placement_map: dict[str, dict[str, Any]],
    default_element_size: int,
) -> int:
    _, access_extent, element_size = _compute_access_window(
        access,
        env=env,
        placement_map=placement_map,
        default_element_size=default_element_size,
    )
    byte_count = int(element_size)
    for extent in access_extent:
        byte_count *= int(extent)
    return int(byte_count)


def _access_element_size(access: TensorAccess, placement: dict[str, Any] | None, default_element_size: int) -> int:
    if placement is not None:
        return int(placement["element_size"])
    dtype = access.attrs.get("dtype")
    if dtype is not None:
        try:
            return dtype_byte_size(dtype)
        except ValueError:
            pass
    return int(default_element_size)


def _is_global_access(access: TensorAccess) -> bool:
    return access.attrs.get("memory_space") == "tensor"


__all__ = [
    "append_general_region_data_placements",
    "append_general_operator_entry",
    "build_data_placement_yaml_payload",
    "build_general_data_placement_list",
    "build_general_task_description",
    "build_general_task_descriptions",
]
