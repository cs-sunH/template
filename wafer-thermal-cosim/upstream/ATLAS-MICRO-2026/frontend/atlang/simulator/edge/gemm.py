"""Edge GEMM cache lookup, placement, and task-description helpers."""

from __future__ import annotations

import contextlib
import copy
import math
import os
from itertools import product
from types import SimpleNamespace
from typing import Any

import yaml
from atlasim import Chip, SimulatorOperatorType

from frontend.util import (
    _GB,
    _KB,
    _MHz,
    get_factors,
    make_contiguous_strides,
    make_dram_task,
    make_tensor_placement,
    partition_list,
    set_pdeathsig,
)

from ..common.layout import align_base_addr, build_layout_record, tensor_volume_bytes
from ...ir.nodes import OpRegion
from ..io import redirect_process_output, run_pool_starmap_interruptible, safe_dump_plain


# ---------------------------------------------------------------------------
# Cache lookup
# ---------------------------------------------------------------------------


def _operator_gemm_signature(operator_entry: dict[str, Any]) -> tuple[int, int, int, int]:
    if "gemm_b" not in operator_entry:
        raise ValueError("GEMM operator entry is missing `gemm_b`.")
    gemm_shape = operator_entry.get("gemm_shape")
    if gemm_shape is None:
        raise ValueError("GEMM operator entry is missing `gemm_shape`.")
    return int(operator_entry["gemm_b"]), int(gemm_shape[0]), int(gemm_shape[1]), int(gemm_shape[2])


def _load_edge_gemm_cache(
    op_name: str,
    operator_entry: dict[str, Any],
    *,
    intermediate_result_dir: str,
    gemm_tiling_cache_dir: str,
) -> dict[str, Any] | None:
    batch_count, m_dim, k_dim, n_dim = _operator_gemm_signature(operator_entry)
    cache_file_name = f"gemm_opt_B{batch_count}_M{m_dim}_K{k_dim}_N{n_dim}_general_layout.yaml"
    cache_base_dir = gemm_tiling_cache_dir if gemm_tiling_cache_dir else intermediate_result_dir
    if not cache_base_dir:
        raise ValueError("Edge extraction requires either `gemm_tiling_cache_dir` or `intermediate_result_dir`.")
    cache_file_path = os.path.join(cache_base_dir, op_name, cache_file_name)
    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as f:
            cached = yaml.safe_load(f)
        if cached is not None:
            return cached
    return None


# ---------------------------------------------------------------------------
# DSE fallback
# ---------------------------------------------------------------------------


def _generate_edge_gemm_data_placements(operator, element_size: int, dram_row_size: int, base_addr: int):
    cur_base_addr = base_addr
    input_placement = make_tensor_placement(
        name=f"input_{operator.name}",
        base_addr=cur_base_addr,
        element_size=int(element_size),
        shape=[int(operator.M), int(operator.K)],
        strides=make_contiguous_strides([int(operator.M), int(operator.K)], last_dim_contiguous=True),
    )
    cur_base_addr += math.ceil(int(operator.M) * int(operator.K) * element_size / dram_row_size) * dram_row_size

    output_placement = make_tensor_placement(
        name=f"output_{operator.name}",
        base_addr=cur_base_addr,
        element_size=int(element_size),
        shape=[int(operator.M), int(operator.N)],
        strides=make_contiguous_strides([int(operator.M), int(operator.N)], last_dim_contiguous=True),
    )
    cur_base_addr += math.ceil(int(operator.M) * int(operator.N) * element_size / dram_row_size) * dram_row_size

    weight_placement_list = []
    for i in range(int(operator.B)):
        weight_placement = make_tensor_placement(
            name=f"weight_{operator.name}_b{i}",
            base_addr=cur_base_addr,
            element_size=int(element_size),
            shape=[int(operator.K), int(operator.N)],
            strides=make_contiguous_strides([int(operator.K), int(operator.N)], last_dim_contiguous=False),
        )
        cur_base_addr += math.ceil(int(operator.K) * int(operator.N) * element_size / dram_row_size) * dram_row_size
        weight_placement_list.append(weight_placement)

    return input_placement, output_placement, weight_placement_list, cur_base_addr


def _edge_explore_gemm_worker(
    input_placement: dict[str, Any],
    output_placement: dict[str, Any],
    weight_placement_list: list[dict[str, Any]],
    operator,
    per_output_accumulate_count: int,
    element_size: int,
    edge_config_dict: dict[str, Any],
    tiling_factor_combination_list: list[tuple[int, int, int]],
    worker_id: int,
    cur_op_dir: str,
    candidate_id: int = 0,
):
    cur_worker_dir = os.path.join(cur_op_dir, f"worker_{worker_id}")
    os.makedirs(cur_worker_dir, exist_ok=True)

    data_placement_config = {"tensor": [input_placement, output_placement, *weight_placement_list]}
    data_placement_config_path = os.path.join(cur_worker_dir, "data_placement.yaml")
    with open(data_placement_config_path, "w") as f:
        safe_dump_plain(data_placement_config, f, sort_keys=False)

    input_name = input_placement["name"]
    output_name = output_placement["name"]
    weight_name = weight_placement_list[0]["name"]

    opt_comp_description = {
        "description": None,
        "tiling_factors": (-1, -1, -1),
        "computation_latency": float("inf"),
        "intra_channel_accumulation_latency": float("inf"),
    }
    opt_latency = float("inf")
    comp_description_path = os.path.join(cur_worker_dir, "comp_description.yaml")
    for tiling_factor_combination in tiling_factor_combination_list:
        tM, tK, tN = tiling_factor_combination
        nM = math.ceil(operator.M / tM)
        nK = math.ceil(operator.K / tK)
        nN = math.ceil(operator.N / tN)
        mac_count = tM * tN * tK
        mac_buffer_load_count = element_size * (tM * tK + tK * tN)
        mac_buffer_store_count = element_size * (tM * tN)
        vec_count = tM * tN
        vec_buffer_load_count = element_size * (tM * tN)
        vec_buffer_store_count = element_size * (tM * tN)

        comp_description = {
            "operator": [
                {
                    "name": operator.name,
                    "type": str(SimulatorOperatorType.GEMM),
                    "iteration": nM * nK * nN,
                    "execution": {
                        "matrix": [{"name": "gemm_tile", "mac_count": mac_count}],
                        "vector": [{"name": "gemm_tile_accumulation", "vec_count": vec_count}],
                        "buffer_load": [
                            {"name": "gemm_tile_load", "byte_count": mac_buffer_load_count, "is_write": False},
                            {
                                "name": "gemm_tile_accumulation_load",
                                "byte_count": vec_buffer_load_count,
                                "is_write": False,
                            },
                        ],
                        "buffer_store": [
                            {"name": "gemm_tile_store", "byte_count": mac_buffer_store_count, "is_write": True},
                            {
                                "name": "gemm_tile_accumulation_store",
                                "byte_count": vec_buffer_store_count,
                                "is_write": True,
                            },
                        ],
                        "dram": [
                            make_dram_task(
                                name=input_name,
                                is_write=False,
                                access_base=[0, 0],
                                access_extent=[tM, tK],
                                access_stride_add=[nK * nN, 1],
                                access_offset_add=[tM, tK],
                                init_iter=0,
                                stride_iter=1 if nK > 1 else nN,
                                total_iter=nM * nK * nN if nK > 1 else nM,
                            ),
                            make_dram_task(
                                name=weight_name,
                                is_write=False,
                                access_base=[0, 0],
                                access_extent=[tK, tN],
                                access_stride_add=[1, nK],
                                access_offset_add=[tK, tN],
                                init_iter=0,
                                stride_iter=1,
                                total_iter=nM * nK * nN if nK * nN > 1 else 1,
                            ),
                            make_dram_task(
                                name=output_name,
                                is_write=True,
                                access_base=[0, 0],
                                access_extent=[tM, tN],
                                access_stride_add=[nK * nN, nK],
                                access_offset_add=[tM, tN],
                                init_iter=nK + 1,
                                stride_iter=nK,
                                total_iter=nM * nN,
                            ),
                        ],
                    },
                }
            ]
        }
        with open(comp_description_path, "w") as f:
            safe_dump_plain(comp_description, f, sort_keys=False)

        log_file_path = os.path.join(cur_worker_dir, "worker_output.log")
        with redirect_process_output(log_file_path):
            with contextlib.suppress(Exception):
                pass
            chip = Chip(edge_config_dict["chip_config_path"], comp_description_path, data_placement_config_path)
            performance = chip.simulate()
            del chip

        computation_latency = performance.e2e_stats.e2e_cycles / (edge_config_dict["frequency"] * _MHz)
        intra_channel_accumulation_latency = 0.0
        intra_channel_accumulation_energy = 0.0
        if per_output_accumulate_count > 0 and edge_config_dict["intra_channel_vector_length"] > 0:
            per_core_output_element_num = operator.M * operator.N
            per_channel_vec_op_count = per_core_output_element_num * per_output_accumulate_count
            accumulation_latency = per_channel_vec_op_count / (
                edge_config_dict["intra_channel_vector_length"] * edge_config_dict["frequency"] * _MHz
            )
            intra_channel_accumulation_latency = accumulation_latency
            intra_channel_accumulation_energy = accumulation_latency * edge_config_dict["vector_power"]

        if computation_latency + intra_channel_accumulation_latency < opt_latency:
            opt_latency = computation_latency + intra_channel_accumulation_latency
            opt_comp_description = {
                "description": comp_description,
                "tiling_factors": (tM, tK, tN),
                "computation_latency": computation_latency,
                "intra_channel_accumulation_latency": intra_channel_accumulation_latency,
                "intra_channel_accumulation_energy": intra_channel_accumulation_energy,
            }

    all_batch_comp_description = {
        "description": [],
        "tiling_factors": (-1, -1, -1),
        "gemm_num": operator.B,
        "computation_latency": float("inf"),
        "intra_channel_accumulation_latency": float("inf"),
        "intra_channel_accumulation_energy": 0.0,
        "latency": float("inf"),
        "candidate_id": candidate_id,
    }
    if opt_comp_description["description"] is not None:
        for b in range(operator.B):
            cur_weight_name = weight_placement_list[b]["name"]
            cur_comp_description = copy.deepcopy(opt_comp_description["description"]["operator"][0])
            cur_comp_description["execution"]["dram"][1]["name"] = cur_weight_name
            all_batch_comp_description["description"].append(cur_comp_description)
        all_batch_comp_description["tiling_factors"] = opt_comp_description["tiling_factors"]
        all_batch_comp_description["computation_latency"] = opt_comp_description["computation_latency"]
        all_batch_comp_description["intra_channel_accumulation_latency"] = opt_comp_description["intra_channel_accumulation_latency"]
        all_batch_comp_description["intra_channel_accumulation_energy"] = opt_comp_description["intra_channel_accumulation_energy"]
        all_batch_comp_description["latency"] = (
            opt_comp_description["computation_latency"] + opt_comp_description["intra_channel_accumulation_latency"]
        )
    return all_batch_comp_description


def _explore_edge_gemm_layout(
    op_name: str,
    operator_entry: dict[str, Any],
    *,
    element_size: int,
    dram_row_size: int,
    system_config,
    intermediate_result_dir: str,
    gemm_tiling_cache_dir: str,
    num_workers: int = 1,
) -> dict[str, int]:
    batch_count, m_dim, k_dim, n_dim = _operator_gemm_signature(operator_entry)
    operator = SimpleNamespace(name=op_name, B=batch_count, M=m_dim, K=k_dim, N=n_dim)
    cache_base_dir = gemm_tiling_cache_dir if gemm_tiling_cache_dir else intermediate_result_dir
    if not cache_base_dir:
        raise ValueError("Edge extraction requires either `gemm_tiling_cache_dir` or `intermediate_result_dir`.")
    cur_op_dir = os.path.join(cache_base_dir, operator.name)
    os.makedirs(cur_op_dir, exist_ok=True)
    cache_file_name = f"gemm_opt_B{batch_count}_M{m_dim}_K{k_dim}_N{n_dim}_general_layout.yaml"
    cache_file_path = os.path.join(cur_op_dir, cache_file_name)

    edge_config_dict = {
        "chip_config_path": system_config.chip_config_path,
        "frequency": system_config.chip_config.frequency,
        "core_num": system_config.chip_config.core_num,
        "intra_channel_vector_length": system_config.intra_channel_vector_length,
        "vector_power": system_config.chip_config.vector_config.power,
    }
    buffer_size = system_config.chip_config.buffer_config.buffer_size * _KB

    channel_num = system_config.channel_num
    channel_num_factor_list = get_factors(channel_num)
    core_num = system_config.chip_config.core_num
    core_num_factor_list = get_factors(core_num) if system_config.intra_channel_vector_length > 0 else [1]

    candidates = []
    for channel_num_factor in channel_num_factor_list:
        if channel_num_factor > operator.B:
            continue
        B_per_group = math.ceil(operator.B / channel_num_factor)
        channel_per_group = math.ceil(channel_num / channel_num_factor)

        for channel_nK in get_factors(channel_per_group):
            channel_nN = channel_per_group // channel_nK
            channel_B = B_per_group
            channel_M = operator.M
            channel_K = math.ceil(operator.K / channel_nK)
            channel_N = math.ceil(operator.N / channel_nN)

            for core_nK in core_num_factor_list:
                core_nN = core_num // core_nK
                core_M = channel_M
                core_K = math.ceil(channel_K / core_nK)
                core_N = math.ceil(channel_N / core_nN)
                core_operator = copy.deepcopy(operator)
                core_operator.B = channel_B
                core_operator.M = core_M
                core_operator.K = core_K
                core_operator.N = core_N

                can_intra_accum = core_nK > 1 and system_config.intra_channel_vector_length > 0
                intra_accum_opts = [True, False] if can_intra_accum else [False]

                for use_intra_accum in intra_accum_opts:
                    per_output_accumulate_count = (core_nK - 1) if use_intra_accum else 0
                    input_scatter_data_volume = channel_B * channel_M * channel_K * element_size
                    output_gather_data_volume = channel_B * channel_M * channel_N * element_size
                    if not use_intra_accum and core_nK > 1:
                        output_gather_data_volume *= core_nK

                    inter_channel_communication_latency = (
                        input_scatter_data_volume + output_gather_data_volume
                    ) / (system_config.channel_bandwidth * _GB)
                    inter_channel_communication_energy = (
                        (input_scatter_data_volume + output_gather_data_volume) * 8 * system_config.channel_energy * 1e-12
                    )

                    if channel_nK > 1 and system_config.npu_vec_length > 0:
                        per_output_accumulate_count_npu = channel_nK - 1
                        npu_vec_op_count = channel_B * channel_M * channel_N * per_output_accumulate_count_npu
                        npu_vec_latency = npu_vec_op_count / (system_config.npu_vec_length * system_config.npu_frequency * _MHz)
                        npu_vec_energy = npu_vec_op_count * system_config.npu_vec_energy * 1e-12
                        inter_channel_communication_latency = max(npu_vec_latency, inter_channel_communication_latency)
                        inter_channel_communication_energy += npu_vec_energy

                    tM_list = get_factors(core_M, 8) or [core_M]
                    cur_min_tK = 128
                    cur_min_tN = 8
                    tK_list = get_factors(core_K, cur_min_tK)
                    while len(tK_list) == 0 and cur_min_tK > 1:
                        cur_min_tK //= 2
                        tK_list = get_factors(core_K, cur_min_tK)
                    tN_list = get_factors(core_N, cur_min_tN)
                    while len(tN_list) == 0 and cur_min_tN > 1:
                        cur_min_tN //= 2
                        tN_list = get_factors(core_N, cur_min_tN)
                    tK_list = tK_list or get_factors(core_K)
                    tN_list = tN_list or get_factors(core_N)
                    legal_tilings = [
                        t for t in product(tM_list, tK_list, tN_list)
                        if element_size * (t[0] * t[1] + t[1] * t[2] + t[0] * t[2]) <= buffer_size / 2
                    ]
                    if not legal_tilings:
                        continue

                    core_input_placement, core_output_placement, core_weight_placement_list, _ = _generate_edge_gemm_data_placements(
                        core_operator, element_size, dram_row_size, 0
                    )
                    candidates.append(
                        {
                            "core_operator": core_operator,
                            "per_output_accumulate_count": per_output_accumulate_count,
                            "legal_tilings": legal_tilings,
                            "channel_B": channel_B,
                            "channel_num_factor": channel_num_factor,
                            "core_nK": core_nK,
                            "core_nN": core_nN,
                            "channel_nK": channel_nK,
                            "channel_nN": channel_nN,
                            "inter_channel_communication_latency": inter_channel_communication_latency,
                            "inter_channel_communication_energy": inter_channel_communication_energy,
                            "core_input_placement": core_input_placement,
                            "core_output_placement": core_output_placement,
                            "core_weight_placement_list": core_weight_placement_list,
                        }
                    )

    if not candidates:
        raise ValueError(f"No legal edge GEMM candidate found for operator '{op_name}'.")

    total_tilings = sum(len(c["legal_tilings"]) for c in candidates)
    work_items = []
    for cand_id, cand in enumerate(candidates):
        workers_for_cand = max(1, round(len(cand["legal_tilings"]) / total_tilings * max(num_workers, 1)))
        partitions = partition_list(cand["legal_tilings"], workers_for_cand)
        for partition in partitions:
            if partition:
                work_items.append(
                    (
                        cand["core_input_placement"],
                        cand["core_output_placement"],
                        cand["core_weight_placement_list"],
                        cand["core_operator"],
                        cand["per_output_accumulate_count"],
                        element_size,
                        edge_config_dict,
                        partition,
                        len(work_items),
                        cur_op_dir,
                        cand_id,
                    )
                )

    worker_count = max(min(max(num_workers, 1), len(work_items)), 1)

    if worker_count <= 1:
        all_results = [_edge_explore_gemm_worker(*item) for item in work_items]
    else:
        all_results = run_pool_starmap_interruptible(
            _edge_explore_gemm_worker,
            work_items,
            processes=worker_count,
            initializer=set_pdeathsig,
            start_method="fork",
        )

    candidate_best: dict[int, dict[str, Any]] = {}
    for result in all_results:
        cand_id = result["candidate_id"]
        if cand_id not in candidate_best or result["latency"] < candidate_best[cand_id]["latency"]:
            candidate_best[cand_id] = result

    best_layout: dict[str, Any] | None = None
    best_latency = float("inf")
    for cand_id, best_result in candidate_best.items():
        cand = candidates[cand_id]
        total_channel_latency = best_result["latency"] * cand["channel_B"]
        total_latency = total_channel_latency + cand["inter_channel_communication_latency"]
        if total_latency < best_latency:
            best_latency = total_latency
            best_layout = {
                "core_M": int(cand["core_operator"].M),
                "core_K": int(cand["core_operator"].K),
                "core_N": int(cand["core_operator"].N),
                "channel_B": int(cand["channel_B"]),
                "description": copy.deepcopy(best_result["description"]),
                "gemm_num_per_channel": int(best_result["gemm_num"]),
                "core_tiling_factors": list(best_result["tiling_factors"]),
                "intra_channel_tiling_factors": [int(cand["core_nK"]), int(cand["core_nN"])],
                "inter_channel_tiling_factors": [int(cand["channel_nK"]), int(cand["channel_nN"])],
                "channel_group_num": int(cand["channel_num_factor"]),
                "computation_latency": float(best_result["computation_latency"] * cand["channel_B"]),
                "intra_channel_accumulation_latency": float(best_result["intra_channel_accumulation_latency"] * cand["channel_B"]),
                "intra_channel_accumulation_energy": float(best_result["intra_channel_accumulation_energy"] * cand["channel_B"]),
                "inter_channel_communication_latency": float(cand["inter_channel_communication_latency"]),
                "inter_channel_communication_energy": float(cand["inter_channel_communication_energy"]),
                "latency": float(total_latency),
            }

    if best_layout is None:
        raise ValueError(f"Failed to finalize edge GEMM exploration for operator '{op_name}'.")

    cache_data = {
        "M": int(operator.M),
        "K": int(operator.K),
        "N": int(operator.N),
        "B": int(operator.B),
        "element_size": int(element_size),
        **best_layout,
    }
    with open(cache_file_path, "w") as f:
        safe_dump_plain(cache_data, f, sort_keys=False)
    return best_layout


def resolve_edge_gemm_cache_data(
    op_name: str,
    operator_entry: dict[str, Any],
    *,
    element_size: int,
    dram_row_size: int,
    system_config,
    intermediate_result_dir: str,
    gemm_tiling_cache_dir: str,
    num_workers: int | None = None,
) -> dict[str, Any]:
    cache_data = _load_edge_gemm_cache(
        op_name,
        operator_entry,
        intermediate_result_dir=intermediate_result_dir,
        gemm_tiling_cache_dir=gemm_tiling_cache_dir,
    )
    if cache_data is None:
        cache_data = _explore_edge_gemm_layout(
            op_name,
            operator_entry,
            element_size=element_size,
            dram_row_size=dram_row_size,
            system_config=system_config,
            intermediate_result_dir=intermediate_result_dir,
            gemm_tiling_cache_dir=gemm_tiling_cache_dir,
            num_workers=max(int(num_workers or 1), 1),
        )
    return cache_data


# ---------------------------------------------------------------------------
# Data placement
# ---------------------------------------------------------------------------


def _build_edge_gemm_layouts(
    op_name: str,
    cache_data: dict[str, Any],
    *,
    base_addr: int,
    element_size: int,
    dram_row_size: int,
) -> tuple[list[dict[str, Any]], int]:
    input_shape = (int(cache_data["core_M"]), int(cache_data["core_K"]))
    output_shape = (int(cache_data["core_M"]), int(cache_data["core_N"]))
    weight_shape = (int(cache_data["core_K"]), int(cache_data["core_N"]))
    input_strides = (input_shape[1], 1)
    output_strides = (output_shape[1], 1)
    weight_strides = (1, weight_shape[0])
    batch_count = int(cache_data["channel_B"])

    placements: list[dict[str, Any]] = []
    input_placement = build_layout_record(f"input_{op_name}", base_addr, input_shape, input_strides, element_size)
    placements.append(input_placement)
    base_addr = align_base_addr(base_addr, tensor_volume_bytes(input_shape, element_size), dram_row_size)

    output_placement = build_layout_record(f"output_{op_name}", base_addr, output_shape, output_strides, element_size)
    placements.append(output_placement)
    base_addr = align_base_addr(base_addr, tensor_volume_bytes(output_shape, element_size), dram_row_size)

    for batch_index in range(batch_count):
        weight_name = f"weight_{op_name}_b{batch_index}"
        weight_placement = build_layout_record(weight_name, base_addr, weight_shape, weight_strides, element_size)
        placements.append(weight_placement)
        base_addr = align_base_addr(base_addr, tensor_volume_bytes(weight_shape, element_size), dram_row_size)

    return placements, base_addr


def build_edge_gemm_region_data_placements(
    op_region: OpRegion,
    *,
    operator_entry: dict[str, Any] | None,
    base_addr: int,
    element_size: int,
    dram_row_size: int,
    system_config,
    intermediate_result_dir: str,
    gemm_tiling_cache_dir: str,
    num_workers: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    if operator_entry is None:
        raise ValueError(f"Operator dict is missing GEMM entry '{op_region.name}'.")
    cache_data = resolve_edge_gemm_cache_data(
        op_region.name,
        operator_entry,
        element_size=element_size,
        dram_row_size=dram_row_size,
        system_config=system_config,
        intermediate_result_dir=intermediate_result_dir,
        gemm_tiling_cache_dir=gemm_tiling_cache_dir,
        num_workers=num_workers,
    )
    return _build_edge_gemm_layouts(
        op_region.name,
        cache_data,
        base_addr=base_addr,
        element_size=element_size,
        dram_row_size=dram_row_size,
    )


# ---------------------------------------------------------------------------
# Task description
# ---------------------------------------------------------------------------

def build_edge_gemm_region_task_description(
    op_region: OpRegion,
    *,
    operator_entry: dict[str, Any] | None,
    element_size: int,
    dram_row_size: int,
    system_config,
    intermediate_result_dir: str,
    gemm_tiling_cache_dir: str,
    num_workers: int | None = None,
) -> tuple[str, dict[str, Any]]:
    if operator_entry is None:
        raise ValueError(f"Operator dict is missing GEMM entry '{op_region.name}'.")
    cache_data = resolve_edge_gemm_cache_data(
        op_region.name,
        operator_entry,
        element_size=element_size,
        dram_row_size=dram_row_size,
        system_config=system_config,
        intermediate_result_dir=intermediate_result_dir,
        gemm_tiling_cache_dir=gemm_tiling_cache_dir,
        num_workers=num_workers,
    )
    return (
        "gemm",
        {
            "name": op_region.name,
            "description": copy.deepcopy(cache_data["description"]),
            "gemm_num_per_channel": int(cache_data["gemm_num_per_channel"]),
            "core_tiling_factors": tuple(cache_data["core_tiling_factors"]),
            "intra_channel_tiling_factors": tuple(cache_data["intra_channel_tiling_factors"]),
            "inter_channel_tiling_factors": tuple(cache_data["inter_channel_tiling_factors"]),
            "channel_group_num": int(cache_data["channel_group_num"]),
            "computation_latency": cache_data.get("computation_latency"),
            "intra_channel_accumulation_latency": cache_data.get("intra_channel_accumulation_latency"),
            "intra_channel_accumulation_energy": cache_data.get("intra_channel_accumulation_energy"),
            "inter_channel_communication_latency": cache_data.get("inter_channel_communication_latency"),
            "inter_channel_communication_energy": cache_data.get("inter_channel_communication_energy"),
            "latency": cache_data.get("latency"),
        },
    )


__all__ = [
    "build_edge_gemm_region_data_placements",
    "build_edge_gemm_region_task_description",
    "resolve_edge_gemm_cache_data",
]
