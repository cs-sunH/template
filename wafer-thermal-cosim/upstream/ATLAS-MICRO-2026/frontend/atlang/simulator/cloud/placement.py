"""Cloud-only data-placement materialization."""

from __future__ import annotations

import copy
from typing import Any

from ..common.general import append_general_region_data_placements
from ..common.layout import align_base_addr, build_layout_record, require_2d_decl, tensor_volume_bytes
from ..common.system import is_edge_system_config
from ...ir.nodes import TensorDecl


# ---------------------------------------------------------------------------
# Fixed cloud operator placement builders
# ---------------------------------------------------------------------------

def _checked_2d_layout(
    tensor_declarations: dict[str, TensorDecl],
    tensor_name: str,
    *,
    expected_layout: str,
) -> tuple[tuple[int, int], tuple[int, int]]:
    tensor_decl = tensor_declarations.get(tensor_name)
    if tensor_decl is None:
        raise ValueError(f"Tensor declaration '{tensor_name}' is missing.")

    shape, strides = require_2d_decl(tensor_decl)
    row_major = strides[1] == 1 and strides[0] == shape[1]
    column_major = strides[0] == 1 and strides[1] == shape[0]
    if expected_layout == "row-major" and not row_major:
        raise ValueError(f"Tensor '{tensor_name}' is expected to be row-major, got shape={shape}, strides={strides}.")
    if expected_layout == "column-major" and not column_major:
        raise ValueError(f"Tensor '{tensor_name}' is expected to be column-major, got shape={shape}, strides={strides}.")
    if expected_layout not in ("row-major", "column-major"):
        raise ValueError(f"Unsupported expected layout {expected_layout!r}.")
    return shape, strides


def _build_cloud_gemm_data_placements(
    op_name: str,
    operator_entry: dict[str, Any] | None,
    tensor_declarations: dict[str, TensorDecl],
    *,
    base_addr: int,
    element_size: int,
    dram_row_size: int,
) -> tuple[list[dict[str, Any]], int]:
    if operator_entry is None:
        raise ValueError(f"Operator dict is missing GEMM entry '{op_name}'.")
    if "gemm_b" not in operator_entry:
        raise ValueError(f"GEMM operator entry '{op_name}' is missing `gemm_b`.")

    placements: list[dict[str, Any]] = []
    input_shape, input_strides = _checked_2d_layout(
        tensor_declarations, f"input_{op_name}", expected_layout="row-major"
    )
    input_placement = build_layout_record(f"input_{op_name}", base_addr, input_shape, input_strides, element_size)
    placements.append(input_placement)
    base_addr = align_base_addr(base_addr, tensor_volume_bytes(input_shape, element_size), dram_row_size)

    output_shape, output_strides = _checked_2d_layout(
        tensor_declarations, f"output_{op_name}", expected_layout="row-major"
    )
    output_placement = build_layout_record(f"output_{op_name}", base_addr, output_shape, output_strides, element_size)
    placements.append(output_placement)
    base_addr = align_base_addr(base_addr, tensor_volume_bytes(output_shape, element_size), dram_row_size)

    weight_shape, weight_strides = _checked_2d_layout(
        tensor_declarations, f"weight_{op_name}", expected_layout="column-major"
    )
    for batch_index in range(int(operator_entry["gemm_b"])):
        weight_placement = build_layout_record(
            f"weight_{op_name}_b{batch_index}", base_addr, weight_shape, weight_strides, element_size
        )
        placements.append(weight_placement)
        base_addr = align_base_addr(base_addr, tensor_volume_bytes(weight_shape, element_size), dram_row_size)

    return placements, base_addr


def _build_cloud_attention_data_placements(
    op_name: str,
    tensor_declarations: dict[str, TensorDecl],
    *,
    base_addr: int,
    element_size: int,
    dram_row_size: int,
) -> tuple[list[dict[str, Any]], int]:
    placements: list[dict[str, Any]] = []
    for tensor_name in (f"input_{op_name}", f"output_{op_name}", "kv_cache"):
        shape, strides = _checked_2d_layout(tensor_declarations, tensor_name, expected_layout="row-major")
        placement = build_layout_record(tensor_name, base_addr, shape, strides, element_size)
        placements.append(placement)
        base_addr = align_base_addr(base_addr, tensor_volume_bytes(shape, element_size), dram_row_size)
    return placements, base_addr


def build_cloud_data_placement_outputs(
    snapshot,
    *,
    operator_dict,
    element_size: int,
    dram_row_size: int,
    system_config,
) -> dict[str, Any]:
    if is_edge_system_config(system_config):
        raise ValueError("build_cloud_data_placement_outputs() only accepts cloud system configs.")

    core_num = int(system_config.chip_config.core_num)
    shared_data_placement_list: list[dict[str, Any]] = []
    per_core_placements: list[list[dict[str, Any]]] = [[] for _ in range(core_num)]
    placement_names_by_core: list[set[str]] = [set() for _ in range(core_num)]
    generated_general_placements: dict[str, dict[str, Any]] = {}
    placed_shared_op_names: set[str] = set()
    base_addr = 0

    for core_array_context in snapshot.core_array_contexts:
        for op_region in core_array_context.op_regions:
            if op_region.op_category in ("gemm", "decode-attention"):
                if op_region.name in placed_shared_op_names:
                    continue
                if op_region.op_category == "gemm":
                    placements, base_addr = _build_cloud_gemm_data_placements(
                        op_region.name,
                        operator_dict.get(op_region.name),
                        snapshot.tensor_declarations,
                        base_addr=base_addr,
                        element_size=element_size,
                        dram_row_size=dram_row_size,
                    )
                elif op_region.op_category == "decode-attention":
                    placements, base_addr = _build_cloud_attention_data_placements(
                        op_region.name,
                        snapshot.tensor_declarations,
                        base_addr=base_addr,
                        element_size=element_size,
                        dram_row_size=dram_row_size,
                    )
                else:
                    raise ValueError(f"Unsupported shared cloud placement category {op_region.op_category!r}.")
                shared_data_placement_list.extend(placements)
                for core_id in range(core_num):
                    per_core_placements[core_id].extend(copy.deepcopy(placements))
                    placement_names_by_core[core_id].update(placement["name"] for placement in placements)
                placed_shared_op_names.add(op_region.name)
            elif op_region.op_category == "communication":
                continue
            elif op_region.op_category == "general":
                base_addr = append_general_region_data_placements(
                    op_region,
                    core_array_context=core_array_context,
                    per_core_placements=per_core_placements,
                    placement_names_by_core=placement_names_by_core,
                    generated_placements=generated_general_placements,
                    tensor_declarations=snapshot.tensor_declarations,
                    base_addr=base_addr,
                    element_size=element_size,
                    dram_row_size=dram_row_size,
                )
            else:
                raise ValueError(f"Unsupported cloud op category {op_region.op_category!r} for placement '{op_region.name}'.")

    return {
        "shared_data_placement_list": shared_data_placement_list,
        "data_placement_list": per_core_placements,
    }


__all__ = ["build_cloud_data_placement_outputs"]
