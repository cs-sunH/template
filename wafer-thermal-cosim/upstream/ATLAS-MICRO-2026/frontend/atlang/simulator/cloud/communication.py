"""Cloud communication task-description extraction driven by AST-captured actions."""

from __future__ import annotations

import copy
import math
import os
from collections import defaultdict
from typing import Any

from atlasim import Chip

from frontend.util import make_dram_task

from ...ir.nodes import KernelRegion, LoopRegion, OpAction, OpRegion
from ..io import redirect_process_output, safe_dump_plain
from ..common.dram import finalize_dram_tasks
from ..common.expressions import evaluate_static_bool, evaluate_static_int, substitute_and_simplify


# -----------------------------------------------------------------------------
# Expression / access evaluation helpers
# -----------------------------------------------------------------------------
def _compute_access_window(
    access,
    *,
    env: dict[str, int],
    placement_map: dict[str, dict[str, Any]],
    default_element_size: int,
) -> tuple[list[int], list[int], int]:
    """Lower one captured access into concrete [base, extent] coordinates."""
    placement = placement_map.get(access.tensor_name)
    if access.slice_shape is None:
        raise ValueError(f"Communication access '{access.tensor_name}' is missing slice_shape.")
    access_base = [
        evaluate_static_int(index_expr, env, description=f"Access base of '{access.tensor_name}' axis {axis}")
        for axis, index_expr in enumerate(access.index_expressions)
    ]
    access_extent = []
    for axis, axis_extent in enumerate(access.slice_shape):
        axis_value = substitute_and_simplify(axis_extent, env)
        if isinstance(axis_value, int):
            access_extent.append(int(axis_value))
            continue
        if hasattr(axis_value, "value") and isinstance(axis_value.value, int):
            access_extent.append(int(axis_value.value))
            continue
        if placement is None:
            raise ValueError(
                f"Communication access '{access.tensor_name}' axis {axis} still contains symbolic extent {axis_value!r} "
                "and no placement is available to resolve it."
            )
        access_extent.append(int(placement["shape"][axis]))
    element_size = default_element_size if placement is None else int(placement["element_size"])
    return access_base, access_extent, element_size


def _access_byte_count(
    access,
    *,
    env: dict[str, int],
    placement_map: dict[str, dict[str, Any]],
    default_element_size: int,
) -> int:
    """Resolve the accessed byte size for one tensor window."""
    access_base, access_extent, element_size = _compute_access_window(
        access,
        env=env,
        placement_map=placement_map,
        default_element_size=default_element_size,
    )
    del access_base
    byte_count = element_size
    for extent in access_extent:
        byte_count *= int(extent)
    return int(byte_count)


def _is_tensor_access(access: Any) -> bool:
    return access.attrs.get("memory_space") == "tensor"


# -----------------------------------------------------------------------------
# Per-core task assembly helpers
# -----------------------------------------------------------------------------
def _empty_on_chip_iteration() -> dict[str, Any]:
    """Allocate one empty on-chip simulator slot for a single core."""
    return {
        "tx": {
            "buffer_load": [],
            "noc": [],
        },
        "rx": {
            "buffer_load": [],
            "buffer_store": [],
            "vector": [],
            "noc": [],
        },
    }


def _ensure_iteration_slot(per_core_description: dict[str, Any], iteration_index: int) -> None:
    """Grow the per-core on-chip timeline until `iteration_index` is addressable."""
    on_chip = per_core_description["communication"]["on_chip"]
    while len(on_chip) <= iteration_index:
        on_chip.append(_empty_on_chip_iteration())


def _append_tx_task(
    *,
    per_core_description: dict[str, Any],
    iteration_index: int,
    src_core: int,
    dst_core: int,
    byte_count: int,
    flit_size: int,
) -> None:
    _ensure_iteration_slot(per_core_description, iteration_index)
    flit_num = math.ceil(byte_count / flit_size)
    tx = per_core_description["communication"]["on_chip"][iteration_index]["tx"]
    tx["buffer_load"].append(
        {
            "name": f"noc_tx{dst_core}_load",
            "byte_count": byte_count,
            "is_write": False,
        }
    )
    tx["noc"].append(
        {
            "name": f"noc_tx{dst_core}",
            "is_send": True,
            "src": src_core,
            "dst": dst_core,
            "flit_num": flit_num,
            "init_flit_id": -1,
        }
    )


def _append_rx_task(
    *,
    per_core_description: dict[str, Any],
    iteration_index: int,
    src_core: int,
    dst_core: int,
    byte_count: int,
    flit_size: int,
) -> None:
    _ensure_iteration_slot(per_core_description, iteration_index)
    flit_num = math.ceil(byte_count / flit_size)
    rx = per_core_description["communication"]["on_chip"][iteration_index]["rx"]
    rx["buffer_store"].append(
        {
            "name": f"noc_rx{src_core}_store",
            "byte_count": byte_count,
            "is_write": True,
        }
    )
    rx["noc"].append(
        {
            "name": f"noc_rx{src_core}",
            "is_send": False,
            "src": src_core,
            "dst": dst_core,
            "flit_num": flit_num,
            "init_flit_id": -1,
        }
    )


def _append_vector_task(
    *,
    per_core_description: dict[str, Any],
    iteration_index: int,
    action: OpAction,
    recv_src_cores: list[int],
    vector_op_per_element: int,
    env: dict[str, int],
    placement_map: dict[str, dict[str, Any]],
    default_element_size: int,
) -> None:
    """Emit the local reduction / elementwise work paired with communication receives."""
    _ensure_iteration_slot(per_core_description, iteration_index)
    rx = per_core_description["communication"]["on_chip"][iteration_index]["rx"]
    input_byte_counts: list[int] = []
    for access in action.input_accesses:
        if _is_tensor_access(access):
            continue
        input_byte_counts.append(
            _access_byte_count(
                access,
                env=env,
                placement_map=placement_map,
                default_element_size=default_element_size,
            )
        )
    if recv_src_cores and input_byte_counts:
        for recv_src_core in sorted(recv_src_cores):
            rx["buffer_load"].append(
                {
                    "name": f"reduce{recv_src_core}_load",
                    "byte_count": sum(input_byte_counts),
                    "is_write": False,
                }
            )
    else:
        for input_index, byte_count in enumerate(input_byte_counts):
            rx["buffer_load"].append(
                {
                    "name": f"{action.attrs.get('op_name', 'vector')}_load_{input_index}",
                    "byte_count": byte_count,
                    "is_write": False,
                }
            )
    output_byte_count = 0
    for output_index, access in enumerate(action.output_accesses):
        if _is_tensor_access(access):
            continue
        byte_count = _access_byte_count(
            access,
            env=env,
            placement_map=placement_map,
            default_element_size=default_element_size,
        )
        if recv_src_cores:
            for recv_src_core in sorted(recv_src_cores):
                rx["buffer_store"].append(
                    {
                        "name": f"reduce{recv_src_core}_store",
                        "byte_count": byte_count,
                        "is_write": True,
                    }
                )
        else:
            rx["buffer_store"].append(
                {
                    "name": f"{action.attrs.get('op_name', 'vector')}_store_{output_index}",
                    "byte_count": byte_count,
                    "is_write": True,
                }
            )
        output_byte_count = max(output_byte_count, byte_count)
    if output_byte_count > 0:
        if recv_src_cores:
            for recv_src_core in sorted(recv_src_cores):
                rx["vector"].append(
                    {
                        "name": f"reduce{recv_src_core}",
                        "vec_count": (output_byte_count // default_element_size) * vector_op_per_element,
                    }
                )
        else:
            rx["vector"].append(
                {
                    "name": action.attrs.get("op_name", "vector"),
                    "vec_count": (output_byte_count // default_element_size) * vector_op_per_element,
                }
            )


def _append_dram_task(
    *,
    per_core_description: dict[str, Any],
    iteration_index: int,
    access,
    is_write: bool,
    env: dict[str, int],
    placement_map: dict[str, dict[str, Any]],
    default_element_size: int,
) -> None:
    """Emit one single-shot DRAM task for a global load/store window."""
    access_base, access_extent, _ = _compute_access_window(
        access,
        env=env,
        placement_map=placement_map,
        default_element_size=default_element_size,
    )
    if any(int(extent) <= 0 for extent in access_extent):
        return
    # Communication DRAM writes land one simulator iteration after the on-chip work
    # that produces the final tensor window, matching the legacy explorer contract.
    dram_init_iter = iteration_index + 1 if is_write else iteration_index
    per_core_description["communication"]["dram"].append(
        make_dram_task(
            name=access.tensor_name,
            is_write=is_write,
            access_base=access_base,
            access_extent=access_extent,
            access_stride_add=[1] * len(access_base),
            access_offset_add=access_extent,
            init_iter=dram_init_iter,
            stride_iter=1,
            total_iter=1,
        )
    )


def _action_is_active(action: OpAction, env: dict[str, int]) -> bool:
    """Evaluate branch predicates attached during builder/simulator_frame capture."""
    conditions = action.attrs.get("conditions", ())
    for index, condition in enumerate(conditions):
        if not evaluate_static_bool(condition, env, description=f"Condition {index} on action '{action.action_kind}'"):
            return False
    return True


# -----------------------------------------------------------------------------
# AST replay engine: region/loop traversal -> communication tasks
# -----------------------------------------------------------------------------
def _emit_action(
    *,
    action: OpAction,
    env: dict[str, int],
    iteration_index: int,
    iteration_state: dict[str, Any],
    vector_op_per_element: int,
    per_core_description: dict[str, Any],
    placement_map: dict[str, dict[str, Any]],
    default_element_size: int,
    flit_size: int,
) -> None:
    """Lower one AST action into communication/vector/DRAM tasks for one iteration slot."""
    if not _action_is_active(action, env):
        return

    if action.action_kind == "tensor_copy":
        global_input = next((access for access in action.input_accesses if _is_tensor_access(access)), None)
        global_output = next((access for access in action.output_accesses if _is_tensor_access(access)), None)
        if global_input is not None and global_output is None:
            _append_dram_task(
                per_core_description=per_core_description,
                iteration_index=iteration_index,
                access=global_input,
                is_write=False,
                env=env,
                placement_map=placement_map,
                default_element_size=default_element_size,
            )
        elif global_output is not None and global_input is None:
            _append_dram_task(
                per_core_description=per_core_description,
                iteration_index=iteration_index,
                access=global_output,
                is_write=True,
                env=env,
                placement_map=placement_map,
                default_element_size=default_element_size,
            )
    elif action.action_kind == "noc_send":
        payload = action.input_accesses[0]
        src_core = evaluate_static_int(action.attrs["src_core"], env, description="noc_send src_core")
        dst_core = evaluate_static_int(action.attrs["dst_core"], env, description="noc_send dst_core")
        byte_count = _access_byte_count(
            payload,
            env=env,
            placement_map=placement_map,
            default_element_size=default_element_size,
        )
        if byte_count <= 0:
            return
        _append_tx_task(
            per_core_description=per_core_description,
            iteration_index=iteration_index,
            src_core=src_core,
            dst_core=dst_core,
            byte_count=byte_count,
            flit_size=flit_size,
        )
    elif action.action_kind == "noc_recv":
        payload = action.output_accesses[0]
        src_core = evaluate_static_int(action.attrs["src_core"], env, description="noc_recv src_core")
        dst_core = evaluate_static_int(action.attrs["dst_core"], env, description="noc_recv dst_core")
        byte_count = _access_byte_count(
            payload,
            env=env,
            placement_map=placement_map,
            default_element_size=default_element_size,
        )
        if byte_count <= 0:
            return
        _append_rx_task(
            per_core_description=per_core_description,
            iteration_index=iteration_index,
            src_core=src_core,
            dst_core=dst_core,
            byte_count=byte_count,
            flit_size=flit_size,
        )
        iteration_state["pending_recv_by_tensor"][payload.tensor_name] = src_core
    elif action.action_kind == "vector_compute":
        recv_src_cores: list[int] = []
        if action.attrs.get("op_name") == "add":
            for access in action.input_accesses:
                recv_src_core = iteration_state["pending_recv_by_tensor"].pop(access.tensor_name, None)
                if recv_src_core is not None:
                    recv_src_cores.append(int(recv_src_core))
            if not recv_src_cores:
                return
        _append_vector_task(
            per_core_description=per_core_description,
            iteration_index=iteration_index,
            action=action,
            recv_src_cores=recv_src_cores,
            vector_op_per_element=vector_op_per_element,
            env=env,
            placement_map=placement_map,
            default_element_size=default_element_size,
        )
    else:
        raise ValueError(f"Unsupported communication action kind {action.action_kind!r}.")


def _execute_loop_region(
    loop_region: LoopRegion,
    *,
    env: dict[str, int],
    current_iter: int,
    iteration_state: dict[str, Any],
    vector_op_per_element: int,
    per_core_description: dict[str, Any],
    placement_map: dict[str, dict[str, Any]],
    default_element_size: int,
    flit_size: int,
) -> tuple[int, bool]:
    """Replay one loop region and advance simulator iterations under current env."""
    extent = evaluate_static_int(loop_region.extent_expression, env, description=f"Loop extent for '{loop_region.loop_variable}'")

    if loop_region.loop_kind == "serial":
        for loop_index in range(extent):
            child_env = dict(env)
            if loop_region.loop_variable is not None:
                child_env[loop_region.loop_variable] = loop_index
            current_iter, slot_used = _execute_region_body(
                loop_region,
                env=child_env,
                current_iter=current_iter,
                iteration_state=iteration_state,
                vector_op_per_element=vector_op_per_element,
                per_core_description=per_core_description,
                placement_map=placement_map,
                default_element_size=default_element_size,
                flit_size=flit_size,
            )
            if slot_used:
                current_iter += 1
                iteration_state["pending_recv_by_tensor"].clear()
        return current_iter, False

    if loop_region.loop_kind not in ("python-range", "general"):
        raise ValueError(f"Unsupported communication loop kind {loop_region.loop_kind!r}.")

    slot_used = False
    for loop_index in range(extent):
        child_env = dict(env)
        if loop_region.loop_variable is not None:
            child_env[loop_region.loop_variable] = loop_index
        current_iter, child_slot_used = _execute_region_body(
            loop_region,
            env=child_env,
            current_iter=current_iter,
            iteration_state=iteration_state,
            vector_op_per_element=vector_op_per_element,
            per_core_description=per_core_description,
            placement_map=placement_map,
            default_element_size=default_element_size,
            flit_size=flit_size,
        )
        slot_used = slot_used or child_slot_used
    return current_iter, slot_used


def _execute_region_body(
    region: KernelRegion | LoopRegion,
    *,
    env: dict[str, int],
    current_iter: int,
    iteration_state: dict[str, Any],
    vector_op_per_element: int,
    per_core_description: dict[str, Any],
    placement_map: dict[str, dict[str, Any]],
    default_element_size: int,
    flit_size: int,
) -> tuple[int, bool]:
    """Replay the ordered body sequence of a kernel/loop region."""
    slot_used = False
    for item_kind, item_index in region.body_sequence:
        if item_kind == "action":
            item = region.actions[item_index]
        elif item_kind == "loop":
            item = region.child_loops[item_index]
        else:
            raise ValueError(f"Unsupported body item kind: {item_kind!r}")

        if item_kind == "action":
            _emit_action(
                action=item,
                env=env,
                iteration_index=current_iter,
                iteration_state=iteration_state,
                vector_op_per_element=vector_op_per_element,
                per_core_description=per_core_description,
                placement_map=placement_map,
                default_element_size=default_element_size,
                flit_size=flit_size,
            )
            if _action_is_active(item, env):
                slot_used = True
        elif item_kind == "loop":
            if item.loop_kind == "serial" and slot_used:
                current_iter += 1
                slot_used = False
                iteration_state["pending_recv_by_tensor"].clear()
            current_iter, child_slot_used = _execute_loop_region(
                item,
                env=env,
                current_iter=current_iter,
                iteration_state=iteration_state,
                vector_op_per_element=vector_op_per_element,
                per_core_description=per_core_description,
                placement_map=placement_map,
                default_element_size=default_element_size,
                flit_size=flit_size,
            )
            if item.loop_kind == "serial":
                pass
            elif item.loop_kind in ("python-range", "general"):
                slot_used = slot_used or child_slot_used
            else:
                raise ValueError(f"Unsupported communication loop kind {item.loop_kind!r}.")
        else:
            raise ValueError(f"Unsupported body item kind: {item_kind!r}")
    return current_iter, slot_used


# -----------------------------------------------------------------------------
# Per-kernel materialization and group assembly
# -----------------------------------------------------------------------------
def _empty_per_core_description() -> dict[str, Any]:
    """Allocate one empty per-core communication description."""
    return {
        "communication": {
            "on_chip": [],
            "dram": [],
        }
    }


def _materialize_kernel_region(
    kernel_region: KernelRegion,
    *,
    core_identifier_name: str | None,
    vector_op_per_element: int,
    placement_map: dict[str, dict[str, Any]],
    default_element_size: int,
    flit_size: int,
) -> tuple[list[dict[str, Any]], int]:
    """Replay one captured kernel for every participating core and materialize per-core YAML."""
    if not kernel_region.core_list:
        raise ValueError("Cloud communication extraction currently requires explicit kernel.core_list.")
    per_core_descriptions = [_empty_per_core_description() for _ in kernel_region.core_list]
    max_iterations = 0
    for local_index, core_id in enumerate(kernel_region.core_list):
        env: dict[str, int] = {}
        if core_identifier_name is not None:
            env[core_identifier_name] = int(core_id)
            env.setdefault("core_id", int(core_id))
        iteration_state = {"pending_recv_by_tensor": {}}
        current_iter, slot_used = _execute_region_body(
            kernel_region,
            env=env,
            current_iter=0,
            iteration_state=iteration_state,
            vector_op_per_element=vector_op_per_element,
            per_core_description=per_core_descriptions[local_index],
            placement_map=placement_map,
            default_element_size=default_element_size,
            flit_size=flit_size,
        )
        if slot_used:
            current_iter += 1
        max_iterations = max(max_iterations, current_iter)
    for per_core_description in per_core_descriptions:
        while len(per_core_description["communication"]["on_chip"]) < max_iterations:
            per_core_description["communication"]["on_chip"].append(_empty_on_chip_iteration())
    return per_core_descriptions, max_iterations


def _merge_group_descriptions(
    *,
    group_name: str,
    core_num: int,
    grouped_pieces: list[tuple[list[dict[str, Any]], int]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Concatenate per-kernel pieces into one operator-level communication description."""
    total_iterations = sum(iteration_count for _, iteration_count in grouped_pieces)
    per_core_descriptions = [_empty_per_core_description() for _ in range(core_num)]
    iter_offset = 0
    for piece_descriptions, piece_iterations in grouped_pieces:
        for core_id, piece_description in enumerate(piece_descriptions):
            per_core_descriptions[core_id]["communication"]["on_chip"].extend(
                copy.deepcopy(piece_description["communication"]["on_chip"])
            )
            for dram_task in piece_description["communication"]["dram"]:
                task_copy = copy.deepcopy(dram_task)
                task_copy["init_iter"] = int(task_copy["init_iter"]) + iter_offset
                per_core_descriptions[core_id]["communication"]["dram"].append(task_copy)
        iter_offset += piece_iterations
    for per_core_description in per_core_descriptions:
        while len(per_core_description["communication"]["on_chip"]) < total_iterations:
            per_core_description["communication"]["on_chip"].append(_empty_on_chip_iteration())
    return (
        {
            "name": group_name,
            "type": "communication",
            "iteration": total_iterations,
            "core_num": core_num,
            "file_prefix": f"{group_name}/",
        },
        per_core_descriptions,
    )


# -----------------------------------------------------------------------------
# Generic DRAM cleanup helpers
# -----------------------------------------------------------------------------
def _eliminate_write_then_read_round_trips(
    dram_tasks: list[dict[str, Any]],
    *,
    tensor_name: str,
) -> list[dict[str, Any]]:
    """Drop redundant same-tile store/load round trips introduced by split communication kernels."""
    writes_by_window: dict[tuple[Any, ...], list[tuple[int, int]]] = defaultdict(list)
    reads_by_window: dict[tuple[Any, ...], list[tuple[int, int]]] = defaultdict(list)
    for task_index, dram_task in enumerate(dram_tasks):
        if dram_task["name"] != tensor_name:
            continue
        entry = (int(dram_task["init_iter"]), task_index)
        window_key = (
            dram_task["name"],
            tuple(int(value) for value in dram_task["access_base"]),
            tuple(int(value) for value in dram_task["access_extent"]),
        )
        if bool(dram_task["is_write"]):
            writes_by_window[window_key].append(entry)
        else:
            reads_by_window[window_key].append(entry)

    removed_indices: set[int] = set()
    for window_key in sorted(set(writes_by_window) | set(reads_by_window)):
        write_entries = sorted(writes_by_window.get(window_key, []))
        read_entries = sorted(reads_by_window.get(window_key, []))
        read_cursor = 0
        for write_iter, write_index in write_entries:
            while read_cursor < len(read_entries) and read_entries[read_cursor][0] < write_iter:
                read_cursor += 1
            if read_cursor >= len(read_entries):
                break
            _, read_index = read_entries[read_cursor]
            removed_indices.add(write_index)
            removed_indices.add(read_index)
            read_cursor += 1
    return [dram_task for task_index, dram_task in enumerate(dram_tasks) if task_index not in removed_indices]


def _eliminate_reads_covered_by_writes(
    dram_tasks: list[dict[str, Any]],
    *,
    tensor_name: str,
) -> list[dict[str, Any]]:
    """Drop redundant reads when the same tile has already been written in the same merged communication flow."""
    earliest_write_iter_by_window: dict[tuple[Any, ...], int] = {}
    for dram_task in dram_tasks:
        if dram_task["name"] != tensor_name or not bool(dram_task["is_write"]):
            continue
        window_key = (
            dram_task["name"],
            tuple(int(value) for value in dram_task["access_base"]),
            tuple(int(value) for value in dram_task["access_extent"]),
        )
        write_iter = int(dram_task["init_iter"])
        previous_iter = earliest_write_iter_by_window.get(window_key)
        if previous_iter is None or write_iter < previous_iter:
            earliest_write_iter_by_window[window_key] = write_iter

    filtered_tasks: list[dict[str, Any]] = []
    for dram_task in dram_tasks:
        if dram_task["name"] == tensor_name and not bool(dram_task["is_write"]):
            earliest_write_iter = earliest_write_iter_by_window.get(
                (
                    dram_task["name"],
                    tuple(int(value) for value in dram_task["access_base"]),
                    tuple(int(value) for value in dram_task["access_extent"]),
                )
            )
            if earliest_write_iter is not None and earliest_write_iter <= int(dram_task["init_iter"]):
                continue
        filtered_tasks.append(dram_task)
    return filtered_tasks


# -----------------------------------------------------------------------------
# ATLAS-specific communication-operator post-process rules
# -----------------------------------------------------------------------------
def _postprocess_qkv_proj_comm_kv_dram(per_core_descriptions: list[dict[str, Any]]) -> None:
    """Apply qkv_proj_comm_kv compatibility fixes after AST pieces are merged."""
    for per_core_description in per_core_descriptions:
        dram_tasks = _eliminate_write_then_read_round_trips(
            per_core_description["communication"]["dram"],
            tensor_name="output_qkv_proj",
        )
        final_kv_write_iter = len(per_core_description["communication"]["on_chip"]) + 1
        normalized_tasks: list[dict[str, Any]] = []
        for dram_task in dram_tasks:
            task_copy = copy.deepcopy(dram_task)
            if task_copy["name"] == "kv_cache" and bool(task_copy["is_write"]):
                task_copy["init_iter"] = final_kv_write_iter
            normalized_tasks.append(task_copy)
        per_core_description["communication"]["dram"] = finalize_dram_tasks(normalized_tasks)


def _postprocess_qkv_proj_comm_q_all_gather_dram(per_core_descriptions: list[dict[str, Any]]) -> None:
    """Drop q_all_gather DRAM loads that are satisfied by q_all_reduce SRAM reuse."""
    for per_core_description in per_core_descriptions:
        per_core_description["communication"]["dram"] = finalize_dram_tasks(
            [
                copy.deepcopy(dram_task)
                for dram_task in per_core_description["communication"]["dram"]
                if not (dram_task["name"] == "input_attention" and not bool(dram_task["is_write"]))
            ]
        )


def _postprocess_attention_comm_attn_all_gather_dram(
    previous_per_core_descriptions: list[dict[str, Any]],
    per_core_descriptions: list[dict[str, Any]],
) -> None:
    """Fuse RS->AG input_o_proj traffic so the merged DRAM timeline matches the legacy contract."""
    for previous_per_core_description, per_core_description in zip(previous_per_core_descriptions, per_core_descriptions):
        retained_previous_tasks: list[dict[str, Any]] = []
        for dram_task in previous_per_core_description["communication"]["dram"]:
            task_copy = copy.deepcopy(dram_task)
            if task_copy["name"] == "input_o_proj" and bool(task_copy["is_write"]):
                continue
            retained_previous_tasks.append(task_copy)
        previous_per_core_description["communication"]["dram"] = finalize_dram_tasks(retained_previous_tasks)
        merged_current_tasks: list[dict[str, Any]] = []
        for dram_task in per_core_description["communication"]["dram"]:
            task_copy = copy.deepcopy(dram_task)
            if task_copy["name"] == "input_o_proj" and not bool(task_copy["is_write"]) and int(task_copy["init_iter"]) == 0:
                task_copy["is_write"] = True
            merged_current_tasks.append(task_copy)
        merged_current_tasks = _eliminate_reads_covered_by_writes(merged_current_tasks, tensor_name="input_o_proj")
        per_core_description["communication"]["dram"] = finalize_dram_tasks(merged_current_tasks)


def _postprocess_group_dram(
    group_name: str,
    per_core_descriptions: list[dict[str, Any]],
    *,
    previous_group_name: str | None = None,
    previous_per_core_descriptions: list[dict[str, Any]] | None = None,
) -> None:
    """Dispatch ATLAS-specific post-merge DRAM fixes by communication operator name."""
    if group_name == "qkv_proj_comm_kv":
        _postprocess_qkv_proj_comm_kv_dram(per_core_descriptions)
    if group_name == "qkv_proj_comm_q_all_gather" and previous_group_name == "qkv_proj_comm_q_all_reduce":
        _postprocess_qkv_proj_comm_q_all_gather_dram(per_core_descriptions)
    if (
        group_name == "attention_comm_attn_all_gather"
        and previous_group_name == "attention_comm_attn_reduce_scatter"
        and previous_per_core_descriptions is not None
    ):
        _postprocess_attention_comm_attn_all_gather_dram(
            previous_per_core_descriptions,
            per_core_descriptions,
        )
    for per_core_description in per_core_descriptions:
        # Generic communication replay can materialize the same tensor-copy DRAM
        # task multiple times when two wave buffers touch the same tile. Collapse
        # these exact duplicates before serializing the final simulator YAML.
        per_core_description["communication"]["dram"] = finalize_dram_tasks(
            [
                copy.deepcopy(dram_task)
                for dram_task in per_core_description["communication"]["dram"]
            ]
        )


# -----------------------------------------------------------------------------
# Final YAML normalization and simulator execution
# -----------------------------------------------------------------------------
def _sort_noc_tasks(per_core_descriptions: list[dict[str, Any]]) -> None:
    """Normalize tx/rx ordering so generated YAML stays deterministic."""
    for per_core_description in per_core_descriptions:
        for iteration in per_core_description["communication"]["on_chip"]:
            tx_noc = iteration["tx"]["noc"]
            tx_buffer_load = iteration["tx"]["buffer_load"]
            if len(tx_noc) == len(tx_buffer_load) and tx_noc:
                tx_pairs = sorted(
                    zip(tx_noc, tx_buffer_load),
                    key=lambda pair: int(pair[0]["dst"]),
                    reverse=True,
                )
                iteration["tx"]["noc"] = [task for task, _ in tx_pairs]
                iteration["tx"]["buffer_load"] = [task for _, task in tx_pairs]

            rx = iteration["rx"]
            reduce_load_by_src = {
                int(task["name"][len("reduce") : -len("_load")]): task
                for task in rx["buffer_load"]
                if task["name"].startswith("reduce") and task["name"].endswith("_load")
            }
            reduce_store_by_src = {
                int(task["name"][len("reduce") : -len("_store")]): task
                for task in rx["buffer_store"]
                if task["name"].startswith("reduce") and task["name"].endswith("_store")
            }
            reduce_vector_by_src = {
                int(task["name"][len("reduce") :]): task
                for task in rx["vector"]
                if task["name"].startswith("reduce")
            }
            noc_by_src = {int(task["src"]): task for task in rx["noc"]}
            noc_store_by_src = {
                int(task["name"][len("noc_rx") : -len("_store")]): task
                for task in rx["buffer_store"]
                if task["name"].startswith("noc_rx") and task["name"].endswith("_store")
            }
            reduce_srcs = sorted(set(reduce_load_by_src) | set(reduce_store_by_src) | set(reduce_vector_by_src) | set(noc_by_src))
            if reduce_srcs and rx["vector"]:
                ordered_buffer_load: list[dict[str, Any]] = []
                ordered_buffer_store: list[dict[str, Any]] = []
                ordered_noc: list[dict[str, Any]] = []
                ordered_vector: list[dict[str, Any]] = []
                for src in reduce_srcs:
                    if src in reduce_load_by_src:
                        ordered_buffer_load.append(reduce_load_by_src[src])
                    if src in noc_store_by_src:
                        ordered_buffer_store.append(noc_store_by_src[src])
                    if src in reduce_store_by_src:
                        ordered_buffer_store.append(reduce_store_by_src[src])
                    if src in noc_by_src:
                        ordered_noc.append(noc_by_src[src])
                    if src in reduce_vector_by_src:
                        ordered_vector.append(reduce_vector_by_src[src])
                rx["buffer_load"] = ordered_buffer_load
                rx["buffer_store"] = ordered_buffer_store
                rx["noc"] = ordered_noc
                rx["vector"] = ordered_vector
                continue

            rx_noc = rx["noc"]
            rx_buffer_store = rx["buffer_store"]
            if len(rx_noc) == len(rx_buffer_store) and rx_noc:
                rx_pairs = sorted(
                    zip(rx_noc, rx_buffer_store),
                    key=lambda pair: int(pair[0]["src"]),
                )
                rx["noc"] = [task for task, _ in rx_pairs]
                rx["buffer_store"] = [task for _, task in rx_pairs]


def _resolve_noc_flit_ids(per_core_descriptions: list[dict[str, Any]], core_flit_index_list: list[int]) -> None:
    """Pair tx/rx tasks and assign stable flit-id ranges in send order."""
    if not per_core_descriptions:
        return
    iteration_count = max(len(per_core_description["communication"]["on_chip"]) for per_core_description in per_core_descriptions)
    for iteration_index in range(iteration_count):
        pending_rx: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for per_core_description in per_core_descriptions:
            on_chip = per_core_description["communication"]["on_chip"]
            if iteration_index >= len(on_chip):
                continue
            for rx_task in on_chip[iteration_index]["rx"]["noc"]:
                pending_rx.setdefault((int(rx_task["src"]), int(rx_task["dst"])), []).append(rx_task)

        for per_core_description in per_core_descriptions:
            on_chip = per_core_description["communication"]["on_chip"]
            if iteration_index >= len(on_chip):
                continue
            for tx_task in on_chip[iteration_index]["tx"]["noc"]:
                key = (int(tx_task["src"]), int(tx_task["dst"]))
                candidate_rx_tasks = pending_rx.get(key)
                if not candidate_rx_tasks:
                    raise ValueError(
                        f"Missing matching noc_recv for communication pair {key} at iteration {iteration_index}."
                    )
                rx_task = candidate_rx_tasks.pop(0)
                if int(tx_task["flit_num"]) != int(rx_task["flit_num"]):
                    raise ValueError(
                        f"Communication pair {key} at iteration {iteration_index} has mismatched flit counts: "
                        f"{tx_task['flit_num']} vs {rx_task['flit_num']}."
                    )
                init_flit_id = core_flit_index_list[key[0]]
                tx_task["init_flit_id"] = init_flit_id
                rx_task["init_flit_id"] = init_flit_id
                core_flit_index_list[key[0]] += int(tx_task["flit_num"])

        for key, leftover_rx_tasks in pending_rx.items():
            if leftover_rx_tasks:
                raise ValueError(f"Unpaired noc_recv tasks remain for communication pair {key} at iteration {iteration_index}.")


def _simulate_communication_operator(
    *,
    operator_description: dict[str, Any],
    per_core_descriptions: list[dict[str, Any]],
    placement_map: dict[str, dict[str, Any]],
    chip_config_path: str,
    intermediate_result_dir: str,
) -> int:
    """Write one operator bundle to disk and run the simulator for latency scoring."""
    cur_op_dir = os.path.join(intermediate_result_dir, operator_description["name"])
    os.makedirs(cur_op_dir, exist_ok=True)

    referenced_tensor_names = {
        dram_task["name"]
        for per_core_description in per_core_descriptions
        for dram_task in per_core_description["communication"]["dram"]
    }
    data_placement_config = {
        "tensor": [copy.deepcopy(placement_map[name]) for name in sorted(referenced_tensor_names)],
    }
    data_placement_config_path = os.path.join(cur_op_dir, "data_placement.yaml")
    with open(data_placement_config_path, "w") as output_file:
        safe_dump_plain(data_placement_config, output_file, sort_keys=False)

    comp_description_path = os.path.join(cur_op_dir, "comp_description.yaml")
    operator_copy = copy.deepcopy(operator_description)
    operator_copy["file_prefix"] = os.path.join(cur_op_dir, operator_copy["file_prefix"])
    os.makedirs(operator_copy["file_prefix"], exist_ok=True)
    with open(comp_description_path, "w") as output_file:
        safe_dump_plain({"operator": [operator_copy]}, output_file, sort_keys=False)

    for core_id, per_core_description in enumerate(per_core_descriptions):
        with open(os.path.join(operator_copy["file_prefix"], f"core_{core_id}.yaml"), "w") as output_file:
            safe_dump_plain(per_core_description, output_file, sort_keys=False)

    log_file_path = os.path.join(cur_op_dir, "operator_output.log")
    with redirect_process_output(log_file_path):
        chip = Chip(chip_config_path, comp_description_path, data_placement_config_path)
        performance = chip.simulate()
        del chip
    return int(performance.e2e_stats.e2e_cycles)


# -----------------------------------------------------------------------------
# Public extraction entry
# -----------------------------------------------------------------------------
def build_cloud_communication_task_descriptions(
    *,
    op_regions: list[OpRegion],
    core_array_kwargs: dict[str, Any],
    data_placement_list: list[dict[str, Any]],
    system_config,
    element_size: int,
    intermediate_result_dir: str,
) -> list[Any]:
    """Build cloud communication task descriptions from AST-captured OpRegion trees."""
    if not op_regions:
        return []

    core_num = int(system_config.chip_config.core_num)
    flit_size = int(system_config.chip_config.noc_config.flit_size)
    vector_op_per_element_by_group = {
        "down_proj_comm_1d_all_reduce": 2,
    }
    placement_map = {placement["name"]: placement for placement in data_placement_list}

    # Group AST regions by the final operator name that should appear in the emitted YAML.
    grouped_op_regions: list[tuple[str, list[OpRegion]]] = []
    current_group_name: str | None = None
    current_group: list[OpRegion] = []
    for op_region in op_regions:
        group_name = op_region.name
        for suffix in ("_reduce_then_scatter", "_gather_then_scatter"):
            if group_name.endswith(suffix):
                group_name = group_name[: -len(suffix)]
                break
        if current_group_name != group_name:
            if current_group:
                grouped_op_regions.append((current_group_name, current_group))
            current_group_name = group_name
            current_group = [op_region]
        else:
            current_group.append(op_region)
    if current_group:
        grouped_op_regions.append((current_group_name, current_group))

    # First materialize each group independently from AST into per-core task descriptions.
    merged_groups: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
    for group_name, grouped_regions in grouped_op_regions:
        grouped_pieces: list[tuple[list[dict[str, Any]], int]] = []
        for op_region in grouped_regions:
            core_identifier_name = op_region.attrs.get("core_identifier")
            group_piece_descriptions = [_empty_per_core_description() for _ in range(core_num)]
            group_piece_iterations = 0
            for kernel_region in op_region.kernel_regions:
                kernel_descriptions, kernel_iterations = _materialize_kernel_region(
                    kernel_region,
                    core_identifier_name=core_identifier_name,
                    vector_op_per_element=vector_op_per_element_by_group.get(group_name, 1),
                    placement_map=placement_map,
                    default_element_size=element_size,
                    flit_size=flit_size,
                )
                for local_index, core_id in enumerate(kernel_region.core_list or []):
                    group_piece_descriptions[int(core_id)] = kernel_descriptions[local_index]
                group_piece_iterations = max(group_piece_iterations, kernel_iterations)
            for per_core_description in group_piece_descriptions:
                while len(per_core_description["communication"]["on_chip"]) < group_piece_iterations:
                    per_core_description["communication"]["on_chip"].append(_empty_on_chip_iteration())
            grouped_pieces.append((group_piece_descriptions, group_piece_iterations))
        operator_description, per_core_descriptions = _merge_group_descriptions(
            group_name=group_name,
            core_num=core_num,
            grouped_pieces=grouped_pieces,
        )
        merged_groups.append((group_name, operator_description, per_core_descriptions))

    # Then run ATLAS-specific cross-group rewrites once neighboring groups are both available.
    previous_group_name = None
    previous_per_core_descriptions: list[dict[str, Any]] | None = None
    for group_name, _, per_core_descriptions in merged_groups:
        _postprocess_group_dram(
            group_name,
            per_core_descriptions,
            previous_group_name=previous_group_name,
            previous_per_core_descriptions=previous_per_core_descriptions,
        )
        previous_group_name = group_name
        previous_per_core_descriptions = per_core_descriptions

    # Finally normalize task order, resolve flit ids, and run the simulator per operator.
    task_descriptions: list[Any] = []
    core_flit_index_list = [0 for _ in range(core_num)]
    for _, operator_description, per_core_descriptions in merged_groups:
        _sort_noc_tasks(per_core_descriptions)
        _resolve_noc_flit_ids(per_core_descriptions, core_flit_index_list)
        latency = _simulate_communication_operator(
            operator_description=operator_description,
            per_core_descriptions=per_core_descriptions,
            placement_map=placement_map,
            chip_config_path=system_config.chip_config_path,
            intermediate_result_dir=os.path.join(intermediate_result_dir, "communication_task_extraction"),
        )
        task_descriptions.append(
            (
                "communication",
                [(operator_description, per_core_descriptions)],
                latency,
            )
        )
    return task_descriptions


__all__ = ["build_cloud_communication_task_descriptions"]
