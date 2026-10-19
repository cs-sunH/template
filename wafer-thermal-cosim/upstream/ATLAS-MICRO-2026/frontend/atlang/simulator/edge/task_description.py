"""Edge task-description materialization in captured operator order."""

from __future__ import annotations

from typing import Any

from ..common.general_autotune import resolve_general_task_description
from ...ir.nodes import SimulatorMetadataSnapshot
from .gemm import build_edge_gemm_region_task_description


# ---------------------------------------------------------------------------
# Task traversal
# ---------------------------------------------------------------------------

def build_edge_task_descriptions(
    snapshot: SimulatorMetadataSnapshot,
    *,
    operator_dict,
    data_placement_list: list[list[dict[str, Any]]],
    element_size: int,
    dram_row_size: int,
    system_config,
    intermediate_result_dir: str,
    gemm_tiling_cache_dir: str,
    num_workers: int | None = None,
) -> list[Any]:
    placement_maps = [{placement["name"]: placement for placement in placements} for placements in data_placement_list]
    task_description_list: list[Any] = []
    for core_array_context in snapshot.core_array_contexts:
        for op_region in core_array_context.op_regions:
            if op_region.op_category == "gemm":
                task_description_list.append(
                    build_edge_gemm_region_task_description(
                        op_region,
                        operator_entry=operator_dict.get(op_region.name),
                        element_size=element_size,
                        dram_row_size=dram_row_size,
                        system_config=system_config,
                        intermediate_result_dir=intermediate_result_dir,
                        gemm_tiling_cache_dir=gemm_tiling_cache_dir,
                        num_workers=num_workers,
                    )
                )
            elif op_region.op_category == "general":
                task_description_list.append(
                    resolve_general_task_description(
                        op_region,
                        core_array_context=core_array_context,
                        system_config=system_config,
                        placement_maps=placement_maps,
                        data_placement_list=data_placement_list,
                        element_size=element_size,
                        intermediate_result_dir=intermediate_result_dir,
                        num_workers=num_workers,
                    )
                )
            else:
                raise ValueError(f"Unsupported edge op category {op_region.op_category!r} for '{op_region.name}'.")
    return task_description_list


__all__ = ["build_edge_task_descriptions"]
