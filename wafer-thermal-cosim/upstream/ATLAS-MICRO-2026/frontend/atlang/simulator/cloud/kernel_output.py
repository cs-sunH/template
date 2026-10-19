"""Cloud kernel output finalization helpers for simulator extraction."""

from __future__ import annotations

import copy
import os
from collections import defaultdict
from typing import Any

from atlasim import Chip

from frontend.model_parser import OperatorType
from frontend.util import _MHz
from ..common.general import append_general_operator_entry, build_data_placement_yaml_payload
from ..common.noc import assign_task_description_noc_flit_ids
from ..io import redirect_process_output, safe_dump_plain


# -----------------------------------------------------------------------------
# Top-level bundle serialization
# -----------------------------------------------------------------------------
def _normalize_inter_chip_communication_list(value: Any) -> list[Any]:
    """Accept the CoreArray kwarg payload and normalize it into a mutable list."""
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
        raise ValueError("Cloud kernel finalization requires a non-empty intermediate_result_dir.")
    os.makedirs(intermediate_result_dir, exist_ok=True)


def _write_data_placement_yaml(
    *,
    data_placement_list: list[list[dict[str, Any]]],
    intermediate_result_dir: str,
) -> str:
    data_placement_config_path = os.path.join(intermediate_result_dir, "data_placement.yaml")
    with open(data_placement_config_path, "w") as output_file:
        safe_dump_plain(build_data_placement_yaml_payload(data_placement_list), output_file, sort_keys=False)
    return data_placement_config_path


def _append_communication_operator_entries(
    *,
    communication_description_list: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    operator_entries: list[dict[str, Any]],
    intermediate_result_dir: str,
    file_prefix_totals: dict[str, int],
    file_prefix_indices: dict[str, int],
) -> None:
    for operator_description, per_core_description_list in communication_description_list:
        operator_copy = copy.deepcopy(operator_description)
        raw_file_prefix = str(operator_copy["file_prefix"]).rstrip("/")
        file_prefix_index = file_prefix_indices[raw_file_prefix]
        file_prefix_indices[raw_file_prefix] += 1
        if file_prefix_totals[raw_file_prefix] > 1:
            unique_file_prefix = f"{raw_file_prefix}/instance{file_prefix_index}"
        else:
            unique_file_prefix = raw_file_prefix
        operator_copy["file_prefix"] = os.path.join(intermediate_result_dir, unique_file_prefix)
        os.makedirs(operator_copy["file_prefix"], exist_ok=True)
        operator_entries.append(operator_copy)

        for core_id, per_core_description in enumerate(per_core_description_list):
            per_core_path = os.path.join(operator_copy["file_prefix"], f"core_{core_id}.yaml")
            with open(per_core_path, "w") as output_file:
                safe_dump_plain(per_core_description, output_file, sort_keys=False)


def write_cloud_top_level_bundle(
    *,
    task_description_list: list[Any],
    data_placement_list: list[list[dict[str, Any]]],
    attn_input: dict[str, Any] | None = None,
    intermediate_result_dir: str,
) -> dict[str, str]:
    """Emit the top-level cloud bundle expected by `cloud.py` and the simulator."""
    _require_output_dir(intermediate_result_dir)
    assign_task_description_noc_flit_ids(task_description_list)

    data_placement_config_path = _write_data_placement_yaml(
        data_placement_list=data_placement_list,
        intermediate_result_dir=intermediate_result_dir,
    )

    file_prefix_totals: dict[str, int] = defaultdict(int)
    for task_description in task_description_list:
        task_kind = task_description[0]
        if task_kind == "communication":
            for operator_description, _ in task_description[1]:
                file_prefix_totals[str(operator_description["file_prefix"]).rstrip("/")] += 1
        elif task_kind in ("computation", "general"):
            pass
        else:
            raise ValueError(f"Unsupported cloud task kind {task_kind!r} while counting communication bundles.")
    file_prefix_indices: dict[str, int] = defaultdict(int)

    attn_input_config_path = os.path.abspath(os.path.join(intermediate_result_dir, "attn_input.yaml"))
    with open(attn_input_config_path, "w") as output_file:
        safe_dump_plain({"attention_input": {} if attn_input is None else attn_input}, output_file, sort_keys=False)

    operator_entries: list[dict[str, Any]] = []
    for task_description in task_description_list:
        task_kind = task_description[0]
        if task_kind == "computation":
            operator_entry = copy.deepcopy(task_description[1])
            if operator_entry.get("type") == "decode-attention":
                operator_entry["attention_input"] = attn_input_config_path
            operator_entries.append(operator_entry)
        elif task_kind == "communication":
            _append_communication_operator_entries(
                communication_description_list=task_description[1],
                operator_entries=operator_entries,
                intermediate_result_dir=intermediate_result_dir,
                file_prefix_totals=file_prefix_totals,
                file_prefix_indices=file_prefix_indices,
            )
        elif task_kind == "general":
            append_general_operator_entry(
                task_payload=task_description[1],
                operator_entries=operator_entries,
                intermediate_result_dir=intermediate_result_dir,
            )
        else:
            raise ValueError(f"Unsupported cloud task kind {task_kind!r} while writing top-level bundle.")

    operator_description_config_path = os.path.join(intermediate_result_dir, "operator_description.yaml")
    with open(operator_description_config_path, "w") as output_file:
        safe_dump_plain({"operator": operator_entries}, output_file, sort_keys=False)

    return {
        "data_placement_config_path": data_placement_config_path,
        "operator_description_config_path": operator_description_config_path,
    }


# -----------------------------------------------------------------------------
# Inter-chip communication estimation
# -----------------------------------------------------------------------------
def estimate_inter_chip_communication_performance(
    *,
    inter_chip_communication_list: list[Any],
    system_config,
    element_size: int,
    batch_size: int = -1,
) -> dict[str, tuple[float, float]]:
    """Port of `CloudSystem.estimate_inter_chip_communication_performance(...)`."""
    performance_dict: dict[str, tuple[float, float]] = {}

    if system_config.tp_scope == "scale_up":
        tp_bandwidth = system_config.interconnect_config.scale_up_bandwidth
        tp_latency = system_config.interconnect_config.scale_up_latency
        tp_energy = system_config.interconnect_config.scale_up_energy
    elif system_config.tp_scope == "scale_out":
        tp_bandwidth = system_config.interconnect_config.scale_out_bandwidth
        tp_latency = system_config.interconnect_config.scale_out_latency
        tp_energy = system_config.interconnect_config.scale_out_energy
    else:
        raise ValueError(f"Unsupported tp_scope {system_config.tp_scope!r}; expected 'scale_up' or 'scale_out'.")

    if system_config.ep_scope == "scale_up":
        ep_bandwidth = system_config.interconnect_config.scale_up_bandwidth
        ep_latency = system_config.interconnect_config.scale_up_latency
        ep_energy = system_config.interconnect_config.scale_up_energy
    elif system_config.ep_scope == "scale_out":
        ep_bandwidth = system_config.interconnect_config.scale_out_bandwidth
        ep_latency = system_config.interconnect_config.scale_out_latency
        ep_energy = system_config.interconnect_config.scale_out_energy
    else:
        raise ValueError(f"Unsupported ep_scope {system_config.ep_scope!r}; expected 'scale_up' or 'scale_out'.")

    for operator in inter_chip_communication_list:
        current_operator = copy.deepcopy(operator)
        if batch_size > 0:
            current_operator.M = batch_size

        if current_operator.op_type == OperatorType.ALLREDUCE:
            per_rank_data_volume = current_operator.get_memory_capacity(element_size)
            total_ar_volume = (
                2 * per_rank_data_volume * (system_config.parallel_config.tp_size - 1) / system_config.parallel_config.tp_size
            )
            latency = max(total_ar_volume / (tp_bandwidth * (2 ** 30)), tp_latency)
            energy = total_ar_volume * 8 * tp_energy / (10 ** 12)
            performance_dict[current_operator.name] = (latency, energy)
        elif current_operator.op_type == OperatorType.ALL2ALL:
            total_data_volume = current_operator.get_memory_capacity(element_size) * system_config.parallel_config.ep_size
            latency = max(total_data_volume / (ep_bandwidth * (2 ** 30)), ep_latency)
            energy = total_data_volume * 8 * ep_energy / (10 ** 12)
            performance_dict[current_operator.name] = (latency, energy)
        else:
            raise ValueError(
                "Cloud inter-chip communication estimation only supports ALLREDUCE / ALL2ALL, "
                f"but got {current_operator.op_type!r} for '{current_operator.name}'."
            )

    return performance_dict


# -----------------------------------------------------------------------------
# Whole-kernel finalization
# -----------------------------------------------------------------------------
def finalize_cloud_kernel_outputs(
    *,
    task_description_list: list[Any],
    data_placement_list: list[list[dict[str, Any]]],
    attn_input: dict[str, Any] | None,
    raw_inter_chip_communication_list: Any,
    system_config,
    num_layers: int,
    element_size: int,
    intermediate_result_dir: str,
) -> dict[str, Any]:
    """Write the cloud bundle, simulate the whole kernel, and aggregate latency fields."""
    inter_chip_communication_list = _normalize_inter_chip_communication_list(raw_inter_chip_communication_list)
    bundle_paths = write_cloud_top_level_bundle(
        task_description_list=task_description_list,
        data_placement_list=data_placement_list,
        attn_input={} if attn_input is None else attn_input,
        intermediate_result_dir=intermediate_result_dir,
    )

    log_file_path = os.path.join(intermediate_result_dir, "kernel_output.log")
    with redirect_process_output(log_file_path):
        chip = Chip(
            arch_config_path=system_config.chip_config_path,
            operator_list_path=bundle_paths["operator_description_config_path"],
            placement_map_path=bundle_paths["data_placement_config_path"],
        )
        intra_chip_computation_performance = chip.simulate()
        del chip

    intra_chip_computation_latency = (
        intra_chip_computation_performance.e2e_stats.e2e_cycles / (system_config.chip_config.frequency * _MHz)
    )
    inter_chip_communication_performance = estimate_inter_chip_communication_performance(
        inter_chip_communication_list=inter_chip_communication_list,
        system_config=system_config,
        element_size=element_size,
    )
    inter_chip_communication_latency = sum(
        performance_value[0] for performance_value in inter_chip_communication_performance.values()
    )
    inter_chip_communication_energy = sum(
        performance_value[1] for performance_value in inter_chip_communication_performance.values()
    )
    chip_config = system_config.chip_config
    core_active_power = (
        chip_config.controller_config.power
        + chip_config.matrix_config.power
        + chip_config.vector_config.power
        + chip_config.buffer_config.power
        + chip_config.dram_config.power
    )
    if chip_config.has_noc:
        core_active_power += chip_config.noc_config.power
    chip_active_power = core_active_power * chip_config.core_num
    latency = (intra_chip_computation_latency + inter_chip_communication_latency) * num_layers
    intra_chip_computation_energy = intra_chip_computation_performance.e2e_stats.e2e_energy
    inter_chip_communication_energy += inter_chip_communication_latency * chip_active_power
    energy = (
        intra_chip_computation_performance.e2e_stats.e2e_energy
        + inter_chip_communication_energy
    ) * num_layers

    return {
        "inter_chip_communication_list": inter_chip_communication_list,
        "operator_description_config_path": bundle_paths["operator_description_config_path"],
        "data_placement_config_path": bundle_paths["data_placement_config_path"],
        "intra_chip_computation_performance": intra_chip_computation_performance,
        "intra_chip_computation_latency": intra_chip_computation_latency,
        "intra_chip_computation_energy": intra_chip_computation_energy,
        "inter_chip_communication_performance": inter_chip_communication_performance,
        "inter_chip_communication_latency": inter_chip_communication_latency,
        "inter_chip_communication_energy": inter_chip_communication_energy,
        "latency": latency,
        "energy": energy,
    }


__all__ = [
    "estimate_inter_chip_communication_performance",
    "finalize_cloud_kernel_outputs",
    "write_cloud_top_level_bundle",
]
