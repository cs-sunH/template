"""Cloud-specific simulator extraction orchestration."""

from __future__ import annotations

from ...ir.dtype import dtype_byte_size
from ..common.system import get_num_layers, get_num_workers, get_primary_context
from .kernel_output import finalize_cloud_kernel_outputs
from .placement import build_cloud_data_placement_outputs
from .task_description import build_cloud_task_description_outputs


def extract_cloud_simulator_outputs(
    snapshot,
    *,
    system_config,
    operator_dict,
    dtype,
    intermediate_result_dir,
    gemm_tiling_cache_dir,
):
    primary_context = get_primary_context(snapshot)
    element_size = dtype_byte_size(dtype)
    dram_row_size = int(primary_context.global_kwargs["dram_row_size"])
    num_workers = get_num_workers(primary_context)
    num_layers = get_num_layers(primary_context)

    extraction_result = snapshot.extraction_result

    # Phase 1: traverse captured operators in execution order and build final per-core placement.
    placement_outputs = build_cloud_data_placement_outputs(
        snapshot,
        operator_dict=operator_dict,
        element_size=element_size,
        dram_row_size=dram_row_size,
        system_config=system_config,
    )
    shared_data_placement_list = placement_outputs["shared_data_placement_list"]
    extraction_result.data_placement_list = placement_outputs["data_placement_list"]

    # Phase 2: traverse captured operators again in execution order and materialize every region.
    task_outputs = build_cloud_task_description_outputs(
        snapshot,
        system_config=system_config,
        operator_dict=operator_dict,
        shared_data_placement_list=shared_data_placement_list,
        data_placement_list=extraction_result.data_placement_list,
        element_size=element_size,
        dram_row_size=dram_row_size,
        intermediate_result_dir=intermediate_result_dir,
        gemm_tiling_cache_dir=gemm_tiling_cache_dir,
        num_workers=num_workers,
    )
    extraction_result.task_description_list = task_outputs["task_description_list"]
    extraction_result.attn_input = task_outputs["attn_input"]
    cloud_kernel_outputs = finalize_cloud_kernel_outputs(
        task_description_list=extraction_result.task_description_list,
        data_placement_list=extraction_result.data_placement_list,
        attn_input=extraction_result.attn_input,
        raw_inter_chip_communication_list=primary_context.global_kwargs.get("inter_chip_communication_list"),
        system_config=system_config,
        num_layers=num_layers,
        element_size=element_size,
        intermediate_result_dir=intermediate_result_dir,
    )

    # Phase 3: copy finalized simulator parse results into the extraction result
    # consumed by JITKernel.extract_simulator_outputs().
    extraction_result.inter_chip_communication_list = cloud_kernel_outputs["inter_chip_communication_list"]
    extraction_result.operator_description_config_path = cloud_kernel_outputs["operator_description_config_path"]
    extraction_result.data_placement_config_path = cloud_kernel_outputs["data_placement_config_path"]
    extraction_result.intra_chip_computation_performance = cloud_kernel_outputs["intra_chip_computation_performance"]
    extraction_result.intra_chip_computation_latency = cloud_kernel_outputs["intra_chip_computation_latency"]
    extraction_result.intra_chip_computation_energy = cloud_kernel_outputs["intra_chip_computation_energy"]
    extraction_result.inter_chip_communication_performance = cloud_kernel_outputs["inter_chip_communication_performance"]
    extraction_result.inter_chip_communication_latency = cloud_kernel_outputs["inter_chip_communication_latency"]
    extraction_result.inter_chip_communication_energy = cloud_kernel_outputs["inter_chip_communication_energy"]
    extraction_result.latency = cloud_kernel_outputs["latency"]
    extraction_result.energy = cloud_kernel_outputs["energy"]

    return extraction_result


__all__ = ["extract_cloud_simulator_outputs"]
