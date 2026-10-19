"""Cloud decode-attention extraction driven by AST-captured actions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from atlasim import SimulatorOperatorType

from frontend.util import _KB

from ..common.scalars import to_python_int
from ...ir.nodes import KernelRegion, LoopRegion, OpAction, OpRegion


@dataclass(slots=True)
class _TensorRoleUsage:
    read_count: int = 0
    write_count: int = 0


@dataclass(slots=True)
class AttentionTensorRoles:
    input_tensor_name: str
    kv_cache_tensor_name: str
    output_tensor_name: str


@dataclass(slots=True)
class AttentionStructure:
    kernel_region: KernelRegion
    innermost_loop: LoopRegion
    data_prep_actions: list[OpAction]
    first_gemm: OpAction
    softmax_actions: list[OpAction]
    second_gemm: OpAction
    accumulation_actions: list[OpAction]
    tensor_roles: AttentionTensorRoles


# ---------------------------------------------------------------------------
# Scalar and attribute normalization
# ---------------------------------------------------------------------------

def _parse_autotune_enabled(kernel_region: KernelRegion) -> bool:
    if kernel_region.autotune_enabled is not None:
        return bool(kernel_region.autotune_enabled)
    autotune_attr = kernel_region.attrs.get("autotune")
    if isinstance(autotune_attr, bool):
        return autotune_attr
    if isinstance(autotune_attr, str):
        return autotune_attr.lower() == "true"
    return False


# ---------------------------------------------------------------------------
# AST traversal helpers
# ---------------------------------------------------------------------------

def _iter_actions_in_loop(loop_region: LoopRegion):
    for item_kind, item_index in loop_region.body_sequence:
        if item_kind == "action":
            yield loop_region.actions[item_index]
        elif item_kind == "loop":
            yield from _iter_actions_in_loop(loop_region.child_loops[item_index])
        else:
            raise ValueError(f"Unsupported decode-attention body item kind {item_kind!r}.")


def _iter_actions_in_kernel(kernel_region: KernelRegion):
    for item_kind, item_index in kernel_region.body_sequence:
        if item_kind == "action":
            yield kernel_region.actions[item_index]
        elif item_kind == "loop":
            yield from _iter_actions_in_loop(kernel_region.child_loops[item_index])
        else:
            raise ValueError(f"Unsupported decode-attention body item kind {item_kind!r}.")


def _first_nested_loop_body(kernel_region: KernelRegion) -> LoopRegion:
    current_loop: LoopRegion | None = None
    current_body_sequence = kernel_region.body_sequence
    current_child_loops = kernel_region.child_loops
    while True:
        next_loop_index = None
        for item_kind, item_index in current_body_sequence:
            if item_kind == "loop":
                next_loop_index = item_index
                break
        if next_loop_index is None:
            if current_loop is None:
                raise ValueError("Decode-attention kernel must contain at least one nested loop block.")
            return current_loop
        current_loop = current_child_loops[next_loop_index]
        current_body_sequence = current_loop.body_sequence
        current_child_loops = current_loop.child_loops


def _ordered_loop_actions(loop_region: LoopRegion) -> list[OpAction]:
    ordered_actions: list[OpAction] = []
    for item_kind, item_index in loop_region.body_sequence:
        if item_kind != "action":
            raise ValueError("Decode-attention innermost loop body must not contain nested loops.")
        ordered_actions.append(loop_region.actions[item_index])
    if not ordered_actions:
        raise ValueError("Decode-attention innermost loop body cannot be empty.")
    return ordered_actions


def _is_local_staging_copy(action: OpAction) -> bool:
    if action.action_kind != "tensor_copy":
        return False
    return all(not _is_tensor_access(access) for access in action.input_accesses + action.output_accesses)


def _is_tensor_access(access: Any) -> bool:
    return access.attrs.get("memory_space") == "tensor"


# ---------------------------------------------------------------------------
# Attention structure validation
# ---------------------------------------------------------------------------

def _collect_global_tensor_roles(kernel_region: KernelRegion) -> dict[str, _TensorRoleUsage]:
    role_usage: dict[str, _TensorRoleUsage] = {}
    for action in _iter_actions_in_kernel(kernel_region):
        for access in action.input_accesses:
            if not _is_tensor_access(access):
                continue
            role_usage.setdefault(access.tensor_name, _TensorRoleUsage()).read_count += 1
        for access in action.output_accesses:
            if not _is_tensor_access(access):
                continue
            role_usage.setdefault(access.tensor_name, _TensorRoleUsage()).write_count += 1
    return role_usage


def _extract_tensor_roles(kernel_region: KernelRegion, data_prep_actions: list[OpAction]) -> AttentionTensorRoles:
    role_usage = _collect_global_tensor_roles(kernel_region)
    if len(role_usage) != 3:
        raise ValueError(
            f"Decode-attention kernel must expose exactly 3 global tensor roles, got {sorted(role_usage.keys())}."
        )

    read_only_names = sorted(
        tensor_name for tensor_name, usage in role_usage.items() if usage.read_count > 0 and usage.write_count == 0
    )
    write_only_names = sorted(
        tensor_name for tensor_name, usage in role_usage.items() if usage.read_count == 0 and usage.write_count > 0
    )
    if len(read_only_names) != 2 or len(write_only_names) != 1:
        raise ValueError(
            "Decode-attention kernel must expose exactly two read-only global tensors and one write-only global tensor."
        )

    kv_cache_candidates = {
        access.tensor_name
        for action in data_prep_actions
        for access in action.input_accesses
        if _is_tensor_access(access)
    }
    if len(kv_cache_candidates) != 1:
        raise ValueError(
            "Decode-attention innermost data-prep copy block must read exactly one global tensor role for KV cache."
        )
    kv_cache_tensor_name = next(iter(kv_cache_candidates))
    if kv_cache_tensor_name not in read_only_names:
        raise ValueError(
            f"Decode-attention KV-cache role '{kv_cache_tensor_name}' is not a read-only global tensor."
        )

    input_tensor_candidates = [tensor_name for tensor_name in read_only_names if tensor_name != kv_cache_tensor_name]
    if len(input_tensor_candidates) != 1:
        raise ValueError("Decode-attention kernel could not uniquely identify the input tensor role.")
    return AttentionTensorRoles(
        input_tensor_name=input_tensor_candidates[0],
        kv_cache_tensor_name=kv_cache_tensor_name,
        output_tensor_name=write_only_names[0],
    )


def _validate_attention_structure(op_region: OpRegion) -> AttentionStructure:
    if len(op_region.kernel_regions) != 1:
        raise ValueError(
            f"Decode-attention region '{op_region.name}' expects exactly one kernel region, got {len(op_region.kernel_regions)}."
        )

    kernel_region = op_region.kernel_regions[0]
    innermost_loop = _first_nested_loop_body(kernel_region)
    ordered_actions = _ordered_loop_actions(innermost_loop)

    # Accept only the fixed semantic layout:
    #   copy* -> gemm -> vector+ -> gemm -> vector+
    action_index = 0
    data_prep_actions: list[OpAction] = []
    while action_index < len(ordered_actions) and ordered_actions[action_index].action_kind == "tensor_copy":
        data_prep_actions.append(ordered_actions[action_index])
        action_index += 1

    if action_index >= len(ordered_actions) or ordered_actions[action_index].action_kind != "matrix_compute":
        raise ValueError("Decode-attention innermost loop body must place the 1st GEMM immediately after data prep.")
    first_gemm = ordered_actions[action_index]
    action_index += 1

    softmax_actions: list[OpAction] = []
    while action_index < len(ordered_actions):
        current_action = ordered_actions[action_index]
        if current_action.action_kind == "vector_compute":
            softmax_actions.append(current_action)
            action_index += 1
        elif _is_local_staging_copy(current_action):
            action_index += 1
        else:
            break
    if not softmax_actions:
        raise ValueError("Decode-attention innermost loop body must contain a non-empty 1st vector block after the 1st GEMM.")

    if action_index >= len(ordered_actions) or ordered_actions[action_index].action_kind != "matrix_compute":
        raise ValueError("Decode-attention innermost loop body must place the 2nd GEMM immediately after the 1st vector block.")
    second_gemm = ordered_actions[action_index]
    action_index += 1

    accumulation_actions: list[OpAction] = []
    while action_index < len(ordered_actions):
        current_action = ordered_actions[action_index]
        if current_action.action_kind == "vector_compute":
            accumulation_actions.append(current_action)
            action_index += 1
        elif _is_local_staging_copy(current_action):
            action_index += 1
        else:
            break
    if not accumulation_actions:
        raise ValueError("Decode-attention innermost loop body must contain a non-empty 2nd vector block after the 2nd GEMM.")

    if action_index != len(ordered_actions):
        raise ValueError(
            "Decode-attention innermost loop body must follow the fixed pattern "
            "`copy -> gemm -> vector block -> gemm -> vector block`."
        )

    return AttentionStructure(
        kernel_region=kernel_region,
        innermost_loop=innermost_loop,
        data_prep_actions=data_prep_actions,
        first_gemm=first_gemm,
        softmax_actions=softmax_actions,
        second_gemm=second_gemm,
        accumulation_actions=accumulation_actions,
        tensor_roles=_extract_tensor_roles(kernel_region, data_prep_actions),
    )


# ---------------------------------------------------------------------------
# Tile/count selection
# ---------------------------------------------------------------------------

def _fixed_token_tile_size(structure: AttentionStructure) -> int:
    for action in structure.data_prep_actions:
        for access in action.input_accesses:
            if not _is_tensor_access(access):
                continue
            if access.tensor_name != structure.tensor_roles.kv_cache_tensor_name:
                continue
            if access.slice_shape is None or len(access.slice_shape) != 2:
                raise ValueError("Decode-attention KV-cache copy must stay rank-2.")
            return to_python_int(access.slice_shape[0], description="Decode-attention fixed token tile size")
    raise ValueError("Decode-attention innermost data-prep block is missing the KV-cache global copy.")


def _autotuned_token_tile_size(
    *,
    structure: AttentionStructure,
    placement_map: dict[str, dict[str, Any]],
    system_config,
    context_slot_mapping: list[dict[int, list[int]]],
    element_size: int,
) -> int:
    kv_cache_placement = placement_map.get(structure.tensor_roles.kv_cache_tensor_name)
    if kv_cache_placement is None:
        raise ValueError(f"Missing placement for decode-attention KV-cache tensor '{structure.tensor_roles.kv_cache_tensor_name}'.")

    kv_vector_length = to_python_int(kv_cache_placement["shape"][1], description="Decode-attention KV vector length")
    kv_vector_size = kv_vector_length * int(element_size)
    sram_buffer_size = int(system_config.chip_config.buffer_config.buffer_size) * _KB
    if sram_buffer_size // 2 < kv_vector_size:
        raise ValueError("Decode-attention autotune requires at least half of the SRAM buffer to fit one KV vector.")

    token_tile_size = max(math.floor((sram_buffer_size / 2) / kv_vector_size), 1)
    for core_mapping in context_slot_mapping:
        for slot_id_list in core_mapping.values():
            token_tile_size = min(token_tile_size, len(slot_id_list))
    return max(int(token_tile_size), 1)


def _select_token_tile_size(
    *,
    op_region: OpRegion,
    structure: AttentionStructure,
    placement_map: dict[str, dict[str, Any]],
    system_config,
    element_size: int,
) -> int:
    context_slot_mapping = op_region.region_kwargs.get("context_slot_mapping")
    if not isinstance(context_slot_mapping, list):
        raise ValueError(f"Decode-attention region '{op_region.name}' is missing a list-like context_slot_mapping.")
    if _parse_autotune_enabled(structure.kernel_region):
        return _autotuned_token_tile_size(
            structure=structure,
            placement_map=placement_map,
            system_config=system_config,
            context_slot_mapping=context_slot_mapping,
            element_size=element_size,
        )
    return _fixed_token_tile_size(structure)


# ---------------------------------------------------------------------------
# Output materialization
# ---------------------------------------------------------------------------

def _build_attention_description(
    *,
    op_region: OpRegion,
    structure: AttentionStructure,
    placement_map: dict[str, dict[str, Any]],
    system_config,
    element_size: int,
) -> dict[str, Any]:
    attention_shape = op_region.attrs.get("bound_values")
    if attention_shape is None:
        attention_shape = op_region.region_kwargs.get("attention_shape")
    if attention_shape is None or len(attention_shape) != 5:
        raise ValueError(f"Decode-attention region '{op_region.name}' is missing a valid attention_shape.")
    _, core_kv_head_num, core_q_head_num_per_kv, _, qk_head_dim = tuple(
        to_python_int(value, description=f"Decode-attention shape axis {axis}")
        for axis, value in enumerate(attention_shape)
    )

    token_tile_size = _select_token_tile_size(
        op_region=op_region,
        structure=structure,
        placement_map=placement_map,
        system_config=system_config,
        element_size=element_size,
    )

    kv_cache_placement = placement_map.get(structure.tensor_roles.kv_cache_tensor_name)
    input_placement = placement_map.get(structure.tensor_roles.input_tensor_name)
    output_placement = placement_map.get(structure.tensor_roles.output_tensor_name)
    if kv_cache_placement is None or input_placement is None or output_placement is None:
        raise ValueError(f"Decode-attention region '{op_region.name}' is missing required placements.")

    # Recover head dimensions from the placement that already encodes concatenated K/V vectors.
    kv_vector_length = to_python_int(kv_cache_placement["shape"][1], description="Decode-attention KV vector length")
    if kv_vector_length <= qk_head_dim:
        raise ValueError(f"Decode-attention KV vector length {kv_vector_length} must exceed qk_head_dim {qk_head_dim}.")
    v_head_dim = kv_vector_length - qk_head_dim

    kv_cache_total_rows = to_python_int(kv_cache_placement["shape"][0], description="Decode-attention KV-cache total rows")
    if kv_cache_total_rows % core_kv_head_num != 0:
        raise ValueError(
            f"Decode-attention KV-cache rows {kv_cache_total_rows} are not divisible by kv_head_num {core_kv_head_num}."
        )
    total_slot_num = kv_cache_total_rows // core_kv_head_num

    softmax_vector_count = len(structure.softmax_actions)
    accumulation_vector_count = len(structure.accumulation_actions)
    q_head_num = core_kv_head_num * core_q_head_num_per_kv

    attention_description = {
        "name": op_region.name,
        "type": str(SimulatorOperatorType.DecodeAttention),
        "iteration": -1,
        "execution": {
            "matrix": [
                {
                    "name": op_region.name + "_qk",
                    "mac_count": core_q_head_num_per_kv * qk_head_dim * token_tile_size,
                },
                {
                    "name": op_region.name + "_sv",
                    "mac_count": core_q_head_num_per_kv * token_tile_size * v_head_dim,
                },
            ],
            "vector": [
                {
                    "name": op_region.name + "_softmax",
                    "vec_count": core_q_head_num_per_kv * token_tile_size * softmax_vector_count,
                },
                {
                    "name": op_region.name + "_accumulation",
                    "vec_count": core_q_head_num_per_kv * v_head_dim * accumulation_vector_count,
                },
            ],
            "buffer_load": [
                {
                    "name": op_region.name + "_qk_load",
                    "byte_count": (core_q_head_num_per_kv * qk_head_dim + qk_head_dim * token_tile_size) * element_size,
                    "is_write": False,
                },
                {
                    "name": op_region.name + "_softmax_load",
                    "byte_count": core_q_head_num_per_kv * token_tile_size * softmax_vector_count * element_size,
                    "is_write": False,
                },
                {
                    "name": op_region.name + "_sv_load",
                    "byte_count": (core_q_head_num_per_kv * token_tile_size + token_tile_size * v_head_dim) * element_size,
                    "is_write": False,
                },
                {
                    "name": op_region.name + "_accumulation_load",
                    "byte_count": core_q_head_num_per_kv * v_head_dim * element_size,
                    "is_write": False,
                },
            ],
            "buffer_store": [
                {
                    "name": op_region.name + "_qk_store",
                    "byte_count": core_q_head_num_per_kv * token_tile_size * element_size,
                    "is_write": True,
                },
                {
                    "name": op_region.name + "_softmax_store",
                    "byte_count": core_q_head_num_per_kv * token_tile_size * element_size,
                    "is_write": True,
                },
                {
                    "name": op_region.name + "_sv_store",
                    "byte_count": core_q_head_num_per_kv * v_head_dim * element_size,
                    "is_write": True,
                },
                {
                    "name": op_region.name + "_accumulation_store",
                    "byte_count": core_q_head_num_per_kv * v_head_dim * element_size,
                    "is_write": True,
                },
            ],
            "dram": [],
        },
    }

    attention_input_template = {
        "q_head_num": q_head_num,
        "kv_head_num": core_kv_head_num,
        "q_head_num_per_kv": core_q_head_num_per_kv,
        "qk_head_dim": qk_head_dim,
        "v_head_dim": v_head_dim,
        "input_tensor_name": input_placement["name"],
        "output_tensor_name": output_placement["name"],
        "kv_cache_tensor_name": kv_cache_placement["name"],
        "total_slot_num": total_slot_num,
        "kv_head_addr_offset": total_slot_num * kv_vector_length * element_size,
        "token_tile_size": token_tile_size,
        "block_size": int(system_config.block_size),
        "total_block_num": math.floor(total_slot_num / int(system_config.block_size)),
        "random": False,
        "random_seed": 0,
        "core_input_list": [],
    }
    return {
        "attention_description": attention_description,
        "attention_input_template": attention_input_template,
    }


def _build_dattn_input_payload(
    attention_input_template: dict[str, Any],
    context_slot_mapping: list[dict[int, list[int]]],
) -> dict[str, Any]:
    token_tile_size = int(attention_input_template["token_tile_size"])
    input_data = {
        "q_head_num": attention_input_template["q_head_num"],
        "kv_head_num": attention_input_template["kv_head_num"],
        "q_head_num_per_kv": attention_input_template["q_head_num_per_kv"],
        "qk_head_dim": attention_input_template["qk_head_dim"],
        "v_head_dim": attention_input_template["v_head_dim"],
        "input_tensor_name": attention_input_template["input_tensor_name"],
        "output_tensor_name": attention_input_template["output_tensor_name"],
        "kv_cache_tensor_name": attention_input_template["kv_cache_tensor_name"],
        "total_slot_num": attention_input_template["total_slot_num"],
        "kv_head_addr_offset": attention_input_template["kv_head_addr_offset"],
        "token_tile_size": token_tile_size,
        "block_size": attention_input_template["block_size"],
        "random": attention_input_template.get("random", False),
        "random_seed": attention_input_template.get("random_seed", 0),
        "core_input_list": [],
    }

    for core_id, core_mapping in enumerate(context_slot_mapping):
        request_indices = list(core_mapping.keys())
        request_tile_count: list[int] = []
        request_tile_info_list: list[list[list[int]]] = []
        for request_index in request_indices:
            slot_id_list = list(core_mapping[request_index])
            request_tile_count.append(math.ceil(len(slot_id_list) / token_tile_size))
            request_tile_info = []
            for tile_start in range(0, len(slot_id_list), token_tile_size):
                request_tile_info.append(slot_id_list[tile_start : tile_start + token_tile_size])
            request_tile_info_list.append(request_tile_info)

        input_data["core_input_list"].append(
            {
                "request_indices": request_indices,
                "request_tile_count": request_tile_count,
                "request_tile_info_list": request_tile_info_list,
            }
        )

    return input_data


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def build_cloud_decode_attention_output(
    *,
    op_region: OpRegion,
    data_placement_list: list[dict[str, Any]],
    system_config,
    element_size: int,
) -> dict[str, Any]:
    structure = _validate_attention_structure(op_region)
    placement_map = {placement["name"]: placement for placement in data_placement_list}
    attention_result = _build_attention_description(
        op_region=op_region,
        structure=structure,
        placement_map=placement_map,
        system_config=system_config,
        element_size=element_size,
    )
    attention_description = attention_result["attention_description"]
    attention_input_template = attention_result["attention_input_template"]

    context_slot_mapping = op_region.region_kwargs.get("context_slot_mapping")
    if not isinstance(context_slot_mapping, list):
        raise ValueError(f"Decode-attention region '{op_region.name}' is missing a list-like context_slot_mapping.")
    attn_input = _build_dattn_input_payload(attention_input_template, context_slot_mapping)
    return {
        "task_description": ("computation", attention_description),
        "attn_input": attn_input,
    }


__all__ = ["build_cloud_decode_attention_output"]
