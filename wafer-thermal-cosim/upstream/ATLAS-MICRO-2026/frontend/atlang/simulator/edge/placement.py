"""Edge data-placement materialization in captured operator order."""

from __future__ import annotations

import copy
from typing import Any

from ..common.general import append_general_region_data_placements
from .gemm import build_edge_gemm_region_data_placements


# ---------------------------------------------------------------------------
# Placement traversal
# ---------------------------------------------------------------------------

def build_edge_data_placement_outputs(
    snapshot,
    *,
    operator_dict,
    element_size: int,
    dram_row_size: int,
    system_config,
    intermediate_result_dir: str,
    gemm_tiling_cache_dir: str,
    num_workers: int | None = None,
) -> dict[str, Any]:
    core_num = int(system_config.chip_config.core_num)
    shared_data_placement_list: list[dict[str, Any]] = []
    per_core_placements: list[list[dict[str, Any]]] = [[] for _ in range(core_num)]
    placement_names_by_core: list[set[str]] = [set() for _ in range(core_num)]
    generated_general_placements: dict[str, dict[str, Any]] = {}
    placed_gemm_names: set[str] = set()
    base_addr = 0

    for core_array_context in snapshot.core_array_contexts:
        for op_region in core_array_context.op_regions:
            if op_region.op_category == "gemm":
                if op_region.name in placed_gemm_names:
                    continue
                placements, base_addr = build_edge_gemm_region_data_placements(
                    op_region,
                    operator_entry=operator_dict.get(op_region.name),
                    base_addr=base_addr,
                    element_size=element_size,
                    dram_row_size=dram_row_size,
                    system_config=system_config,
                    intermediate_result_dir=intermediate_result_dir,
                    gemm_tiling_cache_dir=gemm_tiling_cache_dir,
                    num_workers=num_workers,
                )
                shared_data_placement_list.extend(placements)
                for core_id in range(core_num):
                    per_core_placements[core_id].extend(copy.deepcopy(placements))
                    placement_names_by_core[core_id].update(placement["name"] for placement in placements)
                placed_gemm_names.add(op_region.name)
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
                raise ValueError(f"Unsupported edge op category {op_region.op_category!r} for placement '{op_region.name}'.")

    return {
        "shared_data_placement_list": shared_data_placement_list,
        "data_placement_list": per_core_placements,
    }


__all__ = ["build_edge_data_placement_outputs"]
