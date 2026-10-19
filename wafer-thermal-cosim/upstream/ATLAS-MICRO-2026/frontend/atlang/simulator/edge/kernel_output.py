"""Edge kernel output finalization helpers for simulator extraction."""

from __future__ import annotations

import copy
import os
from typing import Any

from atlasim import Chip

from frontend.util import _MHz
from ..common.general import append_general_operator_entry, build_data_placement_yaml_payload
from ..common.noc import assign_task_description_noc_flit_ids
from ..io import redirect_process_output, safe_dump_plain


# ---------------------------------------------------------------------------
# Top-level bundle serialization
# ---------------------------------------------------------------------------

def _build_data_placement_yaml_payload(data_placement_list: list[list[dict[str, Any]]]) -> dict[str, Any]:
    return build_data_placement_yaml_payload(data_placement_list)


def _normalize_inter_chip_communication_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    raise ValueError(
        "CoreArray kwarg 'inter_chip_communication_list' must be list-like when provided, "
        f"but got {type(value).__name__}."
    )


def _require_output_dir(intermediate_result_dir: str) -> None:
    if not intermediate_result_dir:
        raise ValueError("Edge kernel finalization requires a non-empty intermediate_result_dir.")
    os.makedirs(intermediate_result_dir, exist_ok=True)


def _write_data_placement_yaml(
    *,
    data_placement_list: list[list[dict[str, Any]]],
    intermediate_result_dir: str,
) -> str:
    data_placement_config_path = os.path.join(intermediate_result_dir, "data_placement.yaml")
    with open(data_placement_config_path, "w") as output_file:
        safe_dump_plain(_build_data_placement_yaml_payload(data_placement_list), output_file, sort_keys=False)
    return data_placement_config_path


def _build_edge_operator_entries(task_description_list: list[Any], *, intermediate_result_dir: str) -> dict[str, Any]:
    operator_entries: list[dict[str, Any]] = []
    inter_channel_communication_total_latency = 0.0
    intra_channel_accumulation_total_latency = 0.0
    inter_channel_communication_total_energy = 0.0
    intra_channel_accumulation_total_energy = 0.0
    intra_channel_accumulation_latency_dict: dict[str, float] = {}
    intra_channel_accumulation_energy_dict: dict[str, float] = {}
    inter_channel_communication_latency_dict: dict[str, float] = {}
    inter_channel_communication_energy_dict: dict[str, float] = {}

    for task_kind, task_payload in task_description_list:
        if task_kind == "general":
            append_general_operator_entry(
                task_payload=task_payload,
                operator_entries=operator_entries,
                intermediate_result_dir=intermediate_result_dir,
            )
        elif task_kind == "gemm":
            operator_entries.extend(copy.deepcopy(task_payload["description"]))
            op_name = str(task_payload["name"])
            intra_channel_latency = float(task_payload.get("intra_channel_accumulation_latency") or 0.0)
            inter_channel_latency = float(task_payload.get("inter_channel_communication_latency") or 0.0)
            intra_channel_energy = float(task_payload.get("intra_channel_accumulation_energy") or 0.0)
            inter_channel_energy = float(task_payload.get("inter_channel_communication_energy") or 0.0)
            inter_channel_communication_total_latency += inter_channel_latency
            intra_channel_accumulation_total_latency += intra_channel_latency
            inter_channel_communication_total_energy += inter_channel_energy
            intra_channel_accumulation_total_energy += intra_channel_energy
            intra_channel_accumulation_latency_dict[op_name] = intra_channel_latency
            intra_channel_accumulation_energy_dict[op_name] = intra_channel_energy
            inter_channel_communication_latency_dict[op_name] = inter_channel_latency
            inter_channel_communication_energy_dict[op_name] = inter_channel_energy
        else:
            raise ValueError(f"Unsupported edge task kind {task_kind!r} while writing top-level bundle.")

    return {
        "operator_entries": operator_entries,
        "latency_breakdown": {
            "inter_channel_communication_total_latency": inter_channel_communication_total_latency,
            "intra_channel_accumulation_total_latency": intra_channel_accumulation_total_latency,
            "inter_channel_communication_total_energy": inter_channel_communication_total_energy,
            "intra_channel_accumulation_total_energy": intra_channel_accumulation_total_energy,
            "intra_channel_accumulation_latency_dict": intra_channel_accumulation_latency_dict,
            "intra_channel_accumulation_energy_dict": intra_channel_accumulation_energy_dict,
            "inter_channel_communication_latency_dict": inter_channel_communication_latency_dict,
            "inter_channel_communication_energy_dict": inter_channel_communication_energy_dict,
        },
    }


def write_edge_top_level_bundle(
    *,
    task_description_list: list[Any],
    data_placement_list: list[list[dict[str, Any]]],
    intermediate_result_dir: str = "",
) -> dict[str, Any]:
    _require_output_dir(intermediate_result_dir)
    assign_task_description_noc_flit_ids(task_description_list)

    data_placement_config_path = _write_data_placement_yaml(
        data_placement_list=data_placement_list,
        intermediate_result_dir=intermediate_result_dir,
    )
    operator_result = _build_edge_operator_entries(task_description_list, intermediate_result_dir=intermediate_result_dir)
    operator_description_config_path = os.path.join(intermediate_result_dir, "operator_description.yaml")
    with open(operator_description_config_path, "w") as output_file:
        safe_dump_plain({"operator": operator_result["operator_entries"]}, output_file, sort_keys=False)

    return {
        "data_placement_config_path": data_placement_config_path,
        "operator_description_config_path": operator_description_config_path,
        **operator_result["latency_breakdown"],
    }


# ---------------------------------------------------------------------------
# Edge softmax and whole-kernel finalization
# ---------------------------------------------------------------------------

def estimate_edge_softmax_performance(
    *,
    inter_chip_communication_list: list[Any],
    task_description_list: list[Any],
    data_placement_list: list[list[dict[str, Any]]],
    edge_softmax_metadata: dict[str, Any],
    system_config,
) -> dict[str, float]:
    if not inter_chip_communication_list:
        return {
            "softmax_latency": 0.0,
            "softmax_energy": 0.0,
        }
    if inter_chip_communication_list[0].name != "softmax":
        raise ValueError("Edge softmax latency estimation expects the first inter-chip operator to be 'softmax'.")

    required_keys = ("attention_operator_name", "batch_size", "context_length", "kv_group_num", "kv_head_num")
    missing_keys = [key for key in required_keys if edge_softmax_metadata.get(key) is None]
    if missing_keys:
        raise ValueError(
            "Edge softmax latency estimation requires CoreArray kwargs "
            f"{', '.join('edge_softmax_' + key for key in missing_keys)}."
        )

    attention_operator_name = str(edge_softmax_metadata["attention_operator_name"])
    if attention_operator_name == "":
        raise ValueError("Edge softmax metadata field edge_softmax_attention_operator_name must be non-empty.")

    attention_desc = None
    for task_kind, task_payload in task_description_list:
        if task_kind == "gemm" and task_payload["name"] == attention_operator_name:
            attention_desc = task_payload
            break
    if attention_desc is None:
        raise ValueError(
            "Edge softmax latency estimation requires the configured attention operator "
            f"{attention_operator_name!r} in task_description_list."
        )

    batch_size = int(edge_softmax_metadata["batch_size"])
    context_length = int(edge_softmax_metadata["context_length"])
    kv_group_num = int(edge_softmax_metadata["kv_group_num"])
    kv_head_num = int(edge_softmax_metadata["kv_head_num"])
    if batch_size <= 0 or context_length <= 0 or kv_group_num <= 0 or kv_head_num <= 0:
        raise ValueError(
            "Edge softmax metadata values must be positive, "
            f"but got batch_size={batch_size}, context_length={context_length}, "
            f"kv_group_num={kv_group_num}, kv_head_num={kv_head_num}."
        )

    softmax_latency = float("inf")
    softmax_energy = 0.0
    npu_softmax_op_count = 9
    npu_vec_count = npu_softmax_op_count * kv_head_num * batch_size * kv_group_num * context_length
    npu_latency = npu_vec_count / (system_config.npu_vec_length * system_config.npu_frequency * _MHz)
    npu_energy = npu_vec_count * system_config.npu_vec_energy * 1e-12
    if npu_latency < softmax_latency:
        softmax_latency = npu_latency
        softmax_energy = npu_energy

    intra_vec_len = int(system_config.intra_channel_vector_length)
    if intra_vec_len > 0:
        shared_data_placement_list = data_placement_list[0] if data_placement_list else []
        placement_map = {placement["name"]: placement for placement in shared_data_placement_list}
        output_placement = placement_map.get(f"output_{attention_operator_name}")
        if output_placement is None:
            raise ValueError(f"Missing output placement for '{attention_operator_name}' during edge softmax estimation.")
        channel_B = int(attention_desc["gemm_num_per_channel"])
        core_M = int(output_placement["shape"][0])
        core_N = int(output_placement["shape"][1])
        core_nN = int(attention_desc["intra_channel_tiling_factors"][1])
        channel_N = core_N * core_nN
        per_channel_elements = channel_B * core_M * channel_N
        vc = per_channel_elements * 9
        vec_latency = vc / (intra_vec_len * system_config.chip_config.frequency * _MHz)
        if vec_latency < softmax_latency:
            softmax_latency = vec_latency
            softmax_energy = vec_latency * system_config.chip_config.vector_config.power

    return {
        "softmax_latency": float(softmax_latency),
        "softmax_energy": float(softmax_energy),
    }


def finalize_edge_kernel_outputs(
    *,
    task_description_list: list[Any],
    data_placement_list: list[list[dict[str, Any]]],
    raw_inter_chip_communication_list: Any,
    edge_softmax_metadata: dict[str, Any],
    system_config,
    num_layers: int,
    intermediate_result_dir: str,
) -> dict[str, Any]:
    inter_chip_communication_list = _normalize_inter_chip_communication_list(raw_inter_chip_communication_list)
    bundle_outputs = write_edge_top_level_bundle(
        task_description_list=task_description_list,
        data_placement_list=data_placement_list,
        intermediate_result_dir=intermediate_result_dir,
    )

    log_file_path = os.path.join(intermediate_result_dir, "kernel_output.log")
    with redirect_process_output(log_file_path):
        chip = Chip(
            arch_config_path=system_config.chip_config_path,
            operator_list_path=bundle_outputs["operator_description_config_path"],
            placement_map_path=bundle_outputs["data_placement_config_path"],
        )
        intra_chip_computation_performance = chip.simulate()
        del chip

    intra_chip_computation_latency = (
        intra_chip_computation_performance.e2e_stats.e2e_cycles / (system_config.chip_config.frequency * _MHz)
    )
    softmax_performance = estimate_edge_softmax_performance(
        inter_chip_communication_list=inter_chip_communication_list,
        task_description_list=task_description_list,
        data_placement_list=data_placement_list,
        edge_softmax_metadata=edge_softmax_metadata,
        system_config=system_config,
    )
    softmax_latency = softmax_performance["softmax_latency"]
    softmax_energy = softmax_performance["softmax_energy"]
    latency = (
        intra_chip_computation_latency
        + bundle_outputs["intra_channel_accumulation_total_latency"]
        + bundle_outputs["inter_channel_communication_total_latency"]
        + softmax_latency
    ) * num_layers
    intra_chip_computation_energy = intra_chip_computation_performance.e2e_stats.e2e_energy
    energy = (
        intra_chip_computation_performance.e2e_stats.e2e_energy
        + bundle_outputs["intra_channel_accumulation_total_energy"]
        + bundle_outputs["inter_channel_communication_total_energy"]
        + softmax_energy
    ) * num_layers

    return {
        "inter_chip_communication_list": inter_chip_communication_list,
        "operator_description_config_path": bundle_outputs["operator_description_config_path"],
        "data_placement_config_path": bundle_outputs["data_placement_config_path"],
        "intra_chip_computation_performance": intra_chip_computation_performance,
        "intra_chip_computation_latency": intra_chip_computation_latency,
        "intra_chip_computation_energy": intra_chip_computation_energy,
        "inter_channel_communication_total_latency": bundle_outputs["inter_channel_communication_total_latency"],
        "intra_channel_accumulation_total_latency": bundle_outputs["intra_channel_accumulation_total_latency"],
        "inter_channel_communication_total_energy": bundle_outputs["inter_channel_communication_total_energy"],
        "intra_channel_accumulation_total_energy": bundle_outputs["intra_channel_accumulation_total_energy"],
        "intra_channel_accumulation_latency_dict": bundle_outputs["intra_channel_accumulation_latency_dict"],
        "intra_channel_accumulation_energy_dict": bundle_outputs["intra_channel_accumulation_energy_dict"],
        "inter_channel_communication_latency_dict": bundle_outputs["inter_channel_communication_latency_dict"],
        "inter_channel_communication_energy_dict": bundle_outputs["inter_channel_communication_energy_dict"],
        "softmax_latency": softmax_latency,
        "softmax_energy": softmax_energy,
        "latency": latency,
        "energy": energy,
    }


__all__ = ["estimate_edge_softmax_performance", "finalize_edge_kernel_outputs", "write_edge_top_level_bundle"]
