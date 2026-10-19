"""GEMM task-description extraction helpers for simulator-facing outputs."""

from __future__ import annotations

import copy
from typing import Any

from .attention import build_cloud_decode_attention_output
from .communication import build_cloud_communication_task_descriptions
from .gemm import resolve_cloud_gemm_task_descriptions
from ..common.general_autotune import resolve_general_task_description
from ..common.scalars import positive_int_from_core_array_kwargs, to_python_int
from ...ir.nodes import CoreArrayContext, OpRegion, SimulatorMetadataSnapshot


def _extract_cloud_gemm_task_description(
    op_region: OpRegion,
    operator_entry: dict[str, Any],
    *,
    core_array_kwargs: dict[str, Any],
    data_placement_list: list[dict[str, Any]],
    element_size: int,
    system_config,
    intermediate_result_dir: str,
    gemm_tiling_cache_dir: str,
    num_workers: int | None,
) -> list[Any]:
    placement_map = {placement["name"]: placement for placement in data_placement_list}
    input_name = f"input_{op_region.name}"
    output_name = f"output_{op_region.name}"
    input_placement = placement_map.get(input_name)
    output_placement = placement_map.get(output_name)
    if input_placement is None or output_placement is None:
        raise ValueError(f"Missing cloud GEMM placements for '{op_region.name}'.")

    batch_count = to_python_int(operator_entry["gemm_b"], description=f"{op_region.name}.gemm_b")
    weight_prefix = f"weight_{op_region.name}"
    weight_placement_list: list[dict[str, Any]] = []
    for batch_index in range(batch_count):
        candidate_names = [f"{weight_prefix}_b{batch_index}", weight_prefix]
        for candidate_name in candidate_names:
            placement = placement_map.get(candidate_name)
            if placement is not None and placement not in weight_placement_list:
                weight_placement_list.append(placement)
                break
    if len(weight_placement_list) != batch_count:
        weight_names = [placement["name"] for placement in weight_placement_list]
        raise ValueError(
            f"Cloud GEMM '{op_region.name}' expects {batch_count} weight placements, got {weight_names}."
        )

    gemm_result = resolve_cloud_gemm_task_descriptions(
        op_region=op_region,
        op_name=op_region.name,
        input_placement=input_placement,
        output_placement=output_placement,
        weight_placement_list=weight_placement_list,
        element_size=element_size,
        chip_config=system_config.chip_config,
        chip_config_path=system_config.chip_config_path,
        num_workers=num_workers,
        intermediate_result_dir=intermediate_result_dir,
        gemm_tiling_cache_dir=gemm_tiling_cache_dir,
        min_tile_shape=(
            positive_int_from_core_array_kwargs(core_array_kwargs, "min_tM"),
            positive_int_from_core_array_kwargs(core_array_kwargs, "min_tK"),
            positive_int_from_core_array_kwargs(core_array_kwargs, "min_tN"),
        ),
    )
    latency = gemm_result["latency"]
    comp_description_list = gemm_result["comp_description_list"]
    tiling_factors = gemm_result["tiling_factors"]
    return [("computation", copy.deepcopy(description), tuple(tiling_factors), latency) for description in comp_description_list]


def _normalize_cloud_communication_name(op_region_name: str) -> str:
    for suffix in ("_reduce_then_scatter", "_gather_then_scatter"):
        if op_region_name.endswith(suffix):
            return op_region_name[: -len(suffix)]
    return op_region_name


def _collect_trailing_communication_regions(op_regions: list[OpRegion], start_index: int) -> tuple[list[OpRegion], int]:
    communication_regions: list[OpRegion] = []
    index = start_index
    while index < len(op_regions) and op_regions[index].op_category == "communication":
        communication_regions.append(op_regions[index])
        index += 1
    return communication_regions, index


def _expand_communication_bundles(
    communication_by_name: dict[str, list[Any]],
    communication_names: list[str],
    *,
    repeat_count: int,
) -> list[list[Any]]:
    bundles = [[] for _ in range(repeat_count)]
    if not communication_names:
        return bundles

    for communication_name in communication_names:
        description_queue = communication_by_name.get(communication_name)
        if not description_queue:
            raise ValueError(f"Missing cloud communication description for '{communication_name}'.")

        if repeat_count == 1:
            bundles[0].append(description_queue.pop(0))
        elif len(description_queue) >= repeat_count:
            for bundle_index in range(repeat_count):
                bundles[bundle_index].append(description_queue.pop(0))
        elif len(description_queue) == 1:
            # MoE FFN communication is still described once in AST today. Match
            # the old tiling explorer by replaying an independent lowered
            # operator after each GEMM instance when `B > 1`, while
            # leaving final NoC flit ids to the ordered bundle writer.
            shared_description = description_queue.pop(0)
            for bundle in bundles:
                bundle.append(copy.deepcopy(shared_description))
        else:
            raise ValueError(
                f"Cloud communication '{communication_name}' has {len(description_queue)} descriptions, "
                f"which cannot cover repeat_count={repeat_count}."
            )

    return bundles


def _materialize_communication_run(
    communication_regions: list[OpRegion],
    *,
    core_array_context: CoreArrayContext,
    data_placement_list: list[dict[str, Any]],
    system_config,
    element_size: int,
    intermediate_result_dir: str,
    repeat_count: int,
) -> list[list[Any]]:
    communication_by_name: dict[str, list[Any]] = {}
    for description in build_cloud_communication_task_descriptions(
        op_regions=communication_regions,
        core_array_kwargs=core_array_context.global_kwargs,
        data_placement_list=data_placement_list,
        system_config=system_config,
        element_size=element_size,
        intermediate_result_dir=intermediate_result_dir,
    ):
        communication_name = description[1][0][0]["name"]
        communication_by_name.setdefault(communication_name, []).append(description)
    communication_names: list[str] = []
    previous_name: str | None = None
    for op_region in communication_regions:
        communication_name = _normalize_cloud_communication_name(op_region.name)
        if communication_name == previous_name:
            continue
        communication_names.append(communication_name)
        previous_name = communication_name
    return _expand_communication_bundles(
        communication_by_name,
        communication_names,
        repeat_count=repeat_count,
    )


def build_cloud_task_description_outputs(
    snapshot: SimulatorMetadataSnapshot,
    *,
    system_config,
    operator_dict,
    shared_data_placement_list: list[dict[str, Any]],
    data_placement_list: list[list[dict[str, Any]]],
    element_size: int,
    dram_row_size: int,
    intermediate_result_dir: str,
    gemm_tiling_cache_dir: str,
    num_workers: int | None = None,
) -> dict[str, Any]:
    del dram_row_size
    task_description_list: list[Any] = []
    attn_input: dict[str, Any] | None = None
    placement_maps = [{placement["name"]: placement for placement in placements} for placements in data_placement_list]

    for core_array_context in snapshot.core_array_contexts:
        op_regions = core_array_context.op_regions
        op_index = 0
        while op_index < len(op_regions):
            op_region = op_regions[op_index]
            if op_region.op_category == "gemm":
                operator_entry = operator_dict.get(op_region.name)
                if operator_entry is None:
                    raise ValueError(f"Operator dict is missing GEMM entry '{op_region.name}'.")
                descriptions = _extract_cloud_gemm_task_description(
                    op_region,
                    operator_entry,
                    core_array_kwargs=core_array_context.global_kwargs,
                    data_placement_list=shared_data_placement_list,
                    element_size=element_size,
                    system_config=system_config,
                    intermediate_result_dir=intermediate_result_dir,
                    gemm_tiling_cache_dir=gemm_tiling_cache_dir,
                    num_workers=num_workers,
                )
                if not descriptions:
                    raise ValueError(f"Missing cloud GEMM description for '{op_region.name}'.")
                communication_regions, next_index = _collect_trailing_communication_regions(op_regions, op_index + 1)
                communication_bundles = _materialize_communication_run(
                    communication_regions,
                    core_array_context=core_array_context,
                    data_placement_list=shared_data_placement_list,
                    system_config=system_config,
                    element_size=element_size,
                    intermediate_result_dir=intermediate_result_dir,
                    repeat_count=len(descriptions),
                )
                for bundle_index, description in enumerate(descriptions):
                    task_description_list.append(description)
                    task_description_list.extend(communication_bundles[bundle_index])
                op_index = next_index
            elif op_region.op_category == "decode-attention":
                attention_output = build_cloud_decode_attention_output(
                    op_region=op_region,
                    data_placement_list=shared_data_placement_list,
                    system_config=system_config,
                    element_size=element_size,
                )
                if attn_input is not None:
                    raise ValueError("Cloud extraction currently supports one decode-attention input payload per kernel.")
                attn_input = attention_output["attn_input"]
                task_description_list.append(attention_output["task_description"])
                communication_regions, next_index = _collect_trailing_communication_regions(op_regions, op_index + 1)
                task_description_list.extend(
                    _materialize_communication_run(
                        communication_regions,
                        core_array_context=core_array_context,
                        data_placement_list=shared_data_placement_list,
                        system_config=system_config,
                        element_size=element_size,
                        intermediate_result_dir=intermediate_result_dir,
                        repeat_count=1,
                    )[0]
                )
                op_index = next_index
            elif op_region.op_category == "communication":
                communication_regions, next_index = _collect_trailing_communication_regions(op_regions, op_index)
                task_description_list.extend(
                    _materialize_communication_run(
                        communication_regions,
                        core_array_context=core_array_context,
                        data_placement_list=shared_data_placement_list,
                        system_config=system_config,
                        element_size=element_size,
                        intermediate_result_dir=intermediate_result_dir,
                        repeat_count=1,
                    )[0]
                )
                op_index = next_index
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
                op_index += 1
            else:
                raise ValueError(f"Unsupported cloud op category {op_region.op_category!r} for '{op_region.name}'.")

    return {
        "task_description_list": task_description_list,
        "attn_input": attn_input,
    }


__all__ = [
    "build_cloud_task_description_outputs",
]
