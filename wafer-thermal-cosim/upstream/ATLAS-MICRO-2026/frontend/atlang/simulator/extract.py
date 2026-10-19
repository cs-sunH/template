"""Common atlang snapshot materialization helpers."""

from __future__ import annotations

import os
from typing import Any

from .common.layout import align_base_addr, build_layout_record, normalize_layout_tuple, tensor_volume_bytes
from .common.system import get_primary_context, is_edge_system_config
from ..ir.dtype import dtype_byte_size
from .io import safe_dump_plain, to_yaml_plain
from ..ir.nodes import CoreArrayContext, OpAction, SimulatorExtractionResult, SimulatorMetadataSnapshot, TensorAccess, TensorDecl


# Public extraction dispatch

def extract_simulator_outputs(
    snapshot: SimulatorMetadataSnapshot,
    *,
    system_config,
    operator_dict,
    dtype,
    intermediate_result_dir,
    gemm_tiling_cache_dir,
) -> SimulatorExtractionResult:
    if is_edge_system_config(system_config):
        from .edge.extract import extract_edge_simulator_outputs

        return extract_edge_simulator_outputs(
            snapshot,
            system_config=system_config,
            operator_dict=operator_dict,
            dtype=dtype,
            intermediate_result_dir=intermediate_result_dir,
            gemm_tiling_cache_dir=gemm_tiling_cache_dir,
        )

    from .cloud.extract import extract_cloud_simulator_outputs

    return extract_cloud_simulator_outputs(
        snapshot,
        system_config=system_config,
        operator_dict=operator_dict,
        dtype=dtype,
        intermediate_result_dir=intermediate_result_dir,
        gemm_tiling_cache_dir=gemm_tiling_cache_dir,
    )


# Common snapshot materializer

def materialize_common_snapshot(
    snapshot: SimulatorMetadataSnapshot,
    *,
    output_dir: str | None = None,
    prefix: str = "atlang_common",
) -> SimulatorExtractionResult:
    context = get_primary_context(snapshot)
    output_root = output_dir or context.global_kwargs.get("intermediate_result_dir") or "atlang_outputs"
    os.makedirs(output_root, exist_ok=True)

    data_placement_list = _build_data_placement_list(context)
    task_description_list = _build_task_description_list(context)
    operator_description_config_path = os.path.join(output_root, f"{prefix}_operators.yaml")
    data_placement_config_path = os.path.join(output_root, f"{prefix}_placement.yaml")

    with open(operator_description_config_path, "w") as output_file:
        safe_dump_plain({"operators": task_description_list}, output_file)
    with open(data_placement_config_path, "w") as output_file:
        safe_dump_plain({"data_placement": data_placement_list}, output_file)

    result = snapshot.extraction_result
    result.data_placement_list = data_placement_list
    result.task_description_list = task_description_list
    result.operator_description_config_path = operator_description_config_path
    result.data_placement_config_path = data_placement_config_path
    return result


# Data placement materialization

def _build_data_placement_list(context: CoreArrayContext) -> list[list[dict[str, Any]]]:
    tensor_layouts = _build_tensor_layouts(context)
    return [[dict(layout) for layout in tensor_layouts] for _ in range(context.core_num)]


def _build_tensor_layouts(context: CoreArrayContext) -> list[dict[str, Any]]:
    layouts: list[dict[str, Any]] = []
    base_addr = 0
    dram_row_size = int(context.global_kwargs.get("dram_row_size", 1))
    for tensor_name in sorted(context.tensors):
        tensor_decl = context.tensors[tensor_name]
        if not tensor_decl.has_layout:
            continue
        shape = normalize_layout_tuple(tensor_decl.shape, field_name="shape", tensor_name=tensor_decl.name)
        strides = normalize_layout_tuple(tensor_decl.strides, field_name="strides", tensor_name=tensor_decl.name)
        element_size = _tensor_element_size(tensor_decl, context)
        volume_bytes = tensor_volume_bytes(shape, element_size)
        layouts.append(build_layout_record(tensor_name, base_addr, shape, strides, element_size))
        base_addr = align_base_addr(base_addr, volume_bytes, dram_row_size)
    return layouts


def _tensor_element_size(tensor_decl: TensorDecl, context: CoreArrayContext) -> int:
    if tensor_decl.dtype is not None:
        return dtype_byte_size(tensor_decl.dtype)
    value = context.global_kwargs.get("element_size")
    return int(value) if value is not None else 1


# Operator materialization

def _build_task_description_list(context: CoreArrayContext) -> list[dict[str, Any]]:
    operator_records: list[dict[str, Any]] = []
    for op_region in context.op_regions:
        operator_records.append(
            {
                "name": op_region.name,
                "region_kind": op_region.region_kind,
                "op_category": op_region.op_category,
                "region_kwargs": to_yaml_plain(op_region.region_kwargs),
                "attrs": to_yaml_plain(op_region.attrs),
                "kernels": [
                    {
                        "block_expressions": to_yaml_plain(kernel_region.block_expressions),
                        "autotune_enabled": kernel_region.autotune_enabled,
                        "core_list": kernel_region.core_list,
                        "bound_symbols": list(kernel_region.bound_symbols),
                        "attrs": to_yaml_plain(kernel_region.attrs),
                        "body_sequence": to_yaml_plain(kernel_region.body_sequence),
                        "actions": [_action_record(action) for action in kernel_region.actions],
                        "loops": [_loop_record(loop_region) for loop_region in kernel_region.child_loops],
                    }
                    for kernel_region in op_region.kernel_regions
                ],
            }
        )
    return operator_records


def _loop_record(loop_region) -> dict[str, Any]:
    return {
        "loop_variable": loop_region.loop_variable,
        "extent_expression": to_yaml_plain(loop_region.extent_expression),
        "loop_kind": loop_region.loop_kind,
        "attrs": to_yaml_plain(loop_region.attrs),
        "body_sequence": to_yaml_plain(loop_region.body_sequence),
        "actions": [_action_record(action) for action in loop_region.actions],
        "loops": [_loop_record(child_loop) for child_loop in loop_region.child_loops],
    }


def _action_record(action: OpAction) -> dict[str, Any]:
    return {
        "action_kind": action.action_kind,
        "input_accesses": [_access_record(access) for access in action.input_accesses],
        "output_accesses": [_access_record(access) for access in action.output_accesses],
        "attrs": to_yaml_plain(action.attrs),
    }


def _access_record(access: TensorAccess) -> dict[str, Any]:
    return {
        "tensor_name": access.tensor_name,
        "access_mode": access.access_mode,
        "index_expressions": to_yaml_plain(access.index_expressions),
        "slice_shape": to_yaml_plain(access.slice_shape),
        "slice_strides": to_yaml_plain(access.slice_strides),
        "byte_count": to_yaml_plain(access.byte_count),
        "attrs": to_yaml_plain(access.attrs),
    }


__all__ = ["extract_simulator_outputs", "materialize_common_snapshot"]
