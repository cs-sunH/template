"""Edge-specific simulator extraction orchestration."""

from __future__ import annotations

from ...ir.dtype import dtype_byte_size
from ..common.system import get_num_layers, get_num_workers, get_primary_context
from .kernel_output import finalize_edge_kernel_outputs
from .placement import build_edge_data_placement_outputs
from .task_description import build_edge_task_descriptions


def extract_edge_simulator_outputs(
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
    edge_softmax_metadata = {
        "attention_operator_name": primary_context.global_kwargs.get("edge_softmax_attention_operator_name"),
        "batch_size": primary_context.global_kwargs.get("edge_softmax_batch_size"),
        "context_length": primary_context.global_kwargs.get("edge_softmax_context_length"),
        "kv_group_num": primary_context.global_kwargs.get("edge_softmax_kv_group_num"),
        "kv_head_num": primary_context.global_kwargs.get("edge_softmax_kv_head_num"),
    }

    extraction_result = snapshot.extraction_result

    # Phase 1: traverse captured operators in execution order and build final per-core placement.
    placement_outputs = build_edge_data_placement_outputs(
        snapshot,
        operator_dict=operator_dict,
        element_size=element_size,
        dram_row_size=dram_row_size,
        system_config=system_config,
        intermediate_result_dir=intermediate_result_dir,
        gemm_tiling_cache_dir=gemm_tiling_cache_dir,
        num_workers=num_workers,
    )
    extraction_result.data_placement_list = placement_outputs["data_placement_list"]

    # Phase 2: traverse captured operators again in execution order and materialize every region.
    extraction_result.task_description_list = build_edge_task_descriptions(
        snapshot,
        operator_dict=operator_dict,
        data_placement_list=extraction_result.data_placement_list,
        element_size=element_size,
        dram_row_size=dram_row_size,
        system_config=system_config,
        intermediate_result_dir=intermediate_result_dir,
        gemm_tiling_cache_dir=gemm_tiling_cache_dir,
        num_workers=num_workers,
    )

    edge_kernel_outputs = finalize_edge_kernel_outputs(
        task_description_list=extraction_result.task_description_list,
        data_placement_list=extraction_result.data_placement_list,
        raw_inter_chip_communication_list=primary_context.global_kwargs.get("inter_chip_communication_list"),
        edge_softmax_metadata=edge_softmax_metadata,
        system_config=system_config,
        num_layers=num_layers,
        intermediate_result_dir=intermediate_result_dir,
    )

    # Phase 3: copy finalized simulator parse results into the extraction result
    # consumed by JITKernel.extract_simulator_outputs().
    extraction_result.inter_chip_communication_list = edge_kernel_outputs["inter_chip_communication_list"]
    extraction_result.operator_description_config_path = edge_kernel_outputs["operator_description_config_path"]
    extraction_result.data_placement_config_path = edge_kernel_outputs["data_placement_config_path"]
    extraction_result.intra_chip_computation_performance = edge_kernel_outputs["intra_chip_computation_performance"]
    extraction_result.intra_chip_computation_latency = edge_kernel_outputs["intra_chip_computation_latency"]
    extraction_result.intra_chip_computation_energy = edge_kernel_outputs["intra_chip_computation_energy"]
    extraction_result.inter_channel_communication_total_latency = edge_kernel_outputs["inter_channel_communication_total_latency"]
    extraction_result.intra_channel_accumulation_total_latency = edge_kernel_outputs["intra_channel_accumulation_total_latency"]
    extraction_result.inter_channel_communication_total_energy = edge_kernel_outputs["inter_channel_communication_total_energy"]
    extraction_result.intra_channel_accumulation_total_energy = edge_kernel_outputs["intra_channel_accumulation_total_energy"]
    extraction_result.intra_channel_accumulation_latency_dict = edge_kernel_outputs["intra_channel_accumulation_latency_dict"]
    extraction_result.intra_channel_accumulation_energy_dict = edge_kernel_outputs["intra_channel_accumulation_energy_dict"]
    extraction_result.inter_channel_communication_latency_dict = edge_kernel_outputs["inter_channel_communication_latency_dict"]
    extraction_result.inter_channel_communication_energy_dict = edge_kernel_outputs["inter_channel_communication_energy_dict"]
    extraction_result.softmax_latency = edge_kernel_outputs["softmax_latency"]
    extraction_result.softmax_energy = edge_kernel_outputs["softmax_energy"]
    extraction_result.latency = edge_kernel_outputs["latency"]
    extraction_result.energy = edge_kernel_outputs["energy"]

    return extraction_result


__all__ = ["extract_edge_simulator_outputs"]
