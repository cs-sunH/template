import csv
import math
import multiprocessing
import os
import random
import sys
from typing import Any, Dict, List, Tuple

import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from frontend.hardware_parser import CloudSystemConfig, EdgeSystemConfig
from frontend.model_parser import (
    ModelConfig,
    Operator,
    OperatorType,
    get_layer_operator_list,
    get_model_config_from_hf,
    set_cloud_shape,
    set_edge_shape,
)
from frontend.ops.atlang.cloud import cloud_inference, sanity_check
from frontend.ops.atlang.edge import edge_inference
from frontend.ops.atlang.general_mpmd import general_mpmd_autotune_inference, general_mpmd_fixed_inference
from frontend.ops.atlang.general_spmd import general_spmd_autotune_inference, general_spmd_fixed_inference
from frontend.util import generate_context_slot_mapping, generate_last_slot_mapping


def _write_e2e_performance(
    output_dir: str,
    system: str,
    model_name: str,
    batch_size: int,
    context_length: int,
    num_layers: int,
    latency: float,
    energy: float,
) -> None:
    with open(os.path.join(output_dir, "e2e_performance.csv"), "w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow([
            "frontend",
            "system",
            "model",
            "batch_size",
            "context_length",
            "num_layers",
            "e2e_latency_s",
            "e2e_energy_j",
        ])
        writer.writerow([
            "atlang",
            system,
            model_name,
            batch_size,
            context_length,
            num_layers,
            latency,
            energy,
        ])


def _build_cloud_ast_operator_inputs(
    cloud_config: CloudSystemConfig,
    model_config: ModelConfig,
    context_length_list: List[int],
) -> Tuple[Dict[str, Dict[str, Any]], List[Operator]]:
    core_num = cloud_config.chip_config.core_num
    parallel_config = cloud_config.parallel_config
    attention_block, ffn_moe_block = get_layer_operator_list(
        model_config=model_config,
        parallel_config=parallel_config,
    )

    set_cloud_shape(
        model_config=model_config,
        attention_block=attention_block,
        ffn_moe_block=ffn_moe_block,
        context_length_list=context_length_list,
        parallel_config=parallel_config,
        core_num=core_num,
        block_size=cloud_config.block_size,
    )

    operator_dict: Dict[str, Dict[str, Any]] = {}
    inter_chip_communication_list: List[Operator] = []
    for operator in attention_block + ffn_moe_block:
        if operator.op_type in [OperatorType.ALLREDUCE, OperatorType.ALL2ALL]:
            inter_chip_communication_list.append(operator)
            continue

        assert operator.name not in operator_dict, f"Operator {operator.name} already exists"
        operator_dict[operator.name] = {}
        if operator.op_type == OperatorType.GEMM:
            operator_dict[operator.name]["gemm_b"] = operator.B
            operator_dict[operator.name]["gemm_shape"] = (operator.M, operator.K, operator.N)
        elif operator.op_type == OperatorType.ATTENTION:
            operator_dict[operator.name]["attention_shape"] = (
                sum(operator.input_length),
                operator.kv_head_num,
                operator.kv_group_num,
                math.ceil(cloud_config.max_context_length / core_num),
                operator.head_dim,
            )

    return operator_dict, inter_chip_communication_list


def test_cloud_ast_inference(
    cloud_config: CloudSystemConfig,
    model_config: ModelConfig,
    context_length_list: List[int],
    element_size: int,
    intermediate_result_dir: str = "",
    gemm_tiling_cache_dir: str = "",
    num_workers: int = -1,
):
    del element_size
    sanity_check(cloud_config)

    core_num = cloud_config.chip_config.core_num
    core_array_size = int(math.sqrt(core_num))
    operator_dict, inter_chip_communication_list = _build_cloud_ast_operator_inputs(
        cloud_config=cloud_config,
        model_config=model_config,
        context_length_list=context_length_list,
    )

    if intermediate_result_dir == "":
        intermediate_result_dir = f"kick_the_tires/test_cloud_inference_atlang/{model_config.name}_{core_num}cores"
    os.makedirs(intermediate_result_dir, exist_ok=True)

    worker_count = int(0.8 * multiprocessing.cpu_count()) if num_workers < 0 else num_workers
    context_slot_mapping = generate_context_slot_mapping(cloud_config, context_length_list)
    last_slot_mapping = generate_last_slot_mapping(len(context_length_list), core_array_size, core_num)
    return cloud_inference(
        cloud_config=cloud_config,
        operator_dict=operator_dict,
        inter_chip_communication_list=inter_chip_communication_list,
        context_slot_mapping=context_slot_mapping,
        last_slot_mapping=last_slot_mapping,
        intermediate_result_dir=intermediate_result_dir,
        gemm_tiling_cache_dir=gemm_tiling_cache_dir,
        num_layers=model_config.num_layers,
        num_workers=worker_count,
    )


def test_cloud_ast_all(
    system_config_path: str,
    test_config_list: List[Tuple[str, str, int, int, int]],
    intermediate_result_dir: str = "",
    gemm_tiling_cache_dir: str = "",
    num_workers: int = -1,
):
    with open(system_config_path, "r") as input_file:
        cloud_config_yaml = yaml.load(input_file, Loader=yaml.FullLoader)
    cloud_config = CloudSystemConfig.from_yaml(cloud_config_yaml)

    results = []
    for model_name, model_config_path, element_size, batch_size, context_length in test_config_list:
        context_length_list = [context_length] * batch_size
        model_config = get_model_config_from_hf(model_name, model_config_path)
        output_dir = intermediate_result_dir or (
            f"kick_the_tires/test_cloud_inference_atlang/"
            f"{model_config.name}_{cloud_config.chip_config.core_num}cores"
        )
        kernel = test_cloud_ast_inference(
            cloud_config=cloud_config,
            model_config=model_config,
            context_length_list=context_length_list,
            element_size=element_size,
            intermediate_result_dir=output_dir,
            gemm_tiling_cache_dir=gemm_tiling_cache_dir,
            num_workers=num_workers,
        )
        results.append((kernel, kernel.latency, kernel.energy))
        _write_e2e_performance(
            output_dir=output_dir,
            system="cloud",
            model_name=model_name,
            batch_size=batch_size,
            context_length=context_length,
            num_layers=model_config.num_layers,
            latency=kernel.latency,
            energy=kernel.energy,
        )
        print(
            f"{model_name} batch size {batch_size} context length {context_length} "
            f"total inference latency: {kernel.latency}s, total inference energy: {kernel.energy}J, "
            f"per layer inference latency: {kernel.latency / model_config.num_layers}s, "
            f"per layer inference energy: {kernel.energy / model_config.num_layers}J."
        )
    return results


def _build_edge_ast_operator_inputs(
    model_config: ModelConfig,
    context_length_list: List[int],
) -> Tuple[Dict[str, Dict[str, Any]], List[Operator]]:
    attention_block, ffn_moe_block = get_layer_operator_list(
        model_config=model_config,
        skip_communication=True,
        non_fused_attention=True,
    )

    set_edge_shape(
        model_config=model_config,
        attention_block=attention_block,
        ffn_moe_block=ffn_moe_block,
        context_length_list=context_length_list,
    )

    operator_dict: Dict[str, Dict[str, Any]] = {}
    inter_chip_communication_list: List[Operator] = []
    for operator in attention_block + ffn_moe_block:
        if operator.name == "softmax":
            inter_chip_communication_list.append(operator)
            continue

        assert operator.name not in operator_dict, f"Operator {operator.name} already exists"
        operator_dict[operator.name] = {}
        if operator.op_type == OperatorType.GEMM:
            operator_dict[operator.name]["gemm_b"] = operator.B
            operator_dict[operator.name]["gemm_shape"] = (operator.M, operator.K, operator.N)

    return operator_dict, inter_chip_communication_list


def test_edge_ast_inference(
    edge_config: EdgeSystemConfig,
    model_config: ModelConfig,
    context_length_list: List[int],
    element_size: int,
    intermediate_result_dir: str = "",
    gemm_tiling_cache_dir: str = "",
    num_workers: int = -1,
):
    del element_size

    channel_num = edge_config.channel_num
    core_num = edge_config.chip_config.core_num
    operator_dict, inter_chip_communication_list = _build_edge_ast_operator_inputs(
        model_config=model_config,
        context_length_list=context_length_list,
    )

    if intermediate_result_dir == "":
        intermediate_result_dir = f"kick_the_tires/test_edge_inference_atlang/{model_config.name}_{channel_num}ch_{core_num}cores"
    os.makedirs(intermediate_result_dir, exist_ok=True)

    worker_count = int(0.8 * multiprocessing.cpu_count()) if num_workers < 0 else num_workers
    return edge_inference(
        edge_config=edge_config,
        operator_dict=operator_dict,
        inter_chip_communication_list=inter_chip_communication_list,
        intermediate_result_dir=intermediate_result_dir,
        gemm_tiling_cache_dir=gemm_tiling_cache_dir,
        num_layers=model_config.num_layers,
        num_workers=worker_count,
    )


def test_edge_ast_all(
    system_config_path: str,
    test_config_list: List[Tuple[str, str, int, int, int]],
    intermediate_result_dir: str = "",
    gemm_tiling_cache_dir: str = "",
    num_workers: int = -1,
):
    with open(system_config_path, "r") as input_file:
        edge_config_yaml = yaml.load(input_file, Loader=yaml.FullLoader)
    edge_config = EdgeSystemConfig.from_yaml(edge_config_yaml)

    results = []
    for model_name, model_config_path, element_size, batch_size, context_length in test_config_list:
        context_length_list = [context_length] * batch_size
        model_config = get_model_config_from_hf(model_name, model_config_path)
        output_dir = intermediate_result_dir or (
            f"kick_the_tires/test_edge_inference_atlang/"
            f"{model_config.name}_{edge_config.channel_num}ch_{edge_config.chip_config.core_num}cores"
        )
        kernel = test_edge_ast_inference(
            edge_config=edge_config,
            model_config=model_config,
            context_length_list=context_length_list,
            element_size=element_size,
            intermediate_result_dir=output_dir,
            gemm_tiling_cache_dir=gemm_tiling_cache_dir,
            num_workers=num_workers,
        )
        results.append((kernel, kernel.latency, kernel.energy))
        _write_e2e_performance(
            output_dir=output_dir,
            system="edge",
            model_name=model_name,
            batch_size=batch_size,
            context_length=context_length,
            num_layers=model_config.num_layers,
            latency=kernel.latency,
            energy=kernel.energy,
        )
        print(
            f"{model_name} batch size {batch_size} context length {context_length} "
            f"total inference latency: {kernel.latency}s, total inference energy: {kernel.energy}J, "
            f"per layer inference latency: {kernel.latency / model_config.num_layers}s, "
            f"per layer inference energy: {kernel.energy / model_config.num_layers}J."
        )
    return results


def test_general_atlang_samples(
    system_config_path: str,
    intermediate_result_dir: str = "",
    gemm_tiling_cache_dir: str = "",
    num_workers: int = -1,
):
    with open(system_config_path, "r") as input_file:
        cloud_config_yaml = yaml.load(input_file, Loader=yaml.FullLoader)
    cloud_config = CloudSystemConfig.from_yaml(cloud_config_yaml)
    sanity_check(cloud_config)

    worker_count = int(0.8 * multiprocessing.cpu_count()) if num_workers < 0 else num_workers
    output_root = intermediate_result_dir or "decaparated/test_general_atlang"
    os.makedirs(output_root, exist_ok=True)

    spmd_autotune_kernel = general_spmd_autotune_inference(
        cloud_config=cloud_config,
        intermediate_result_dir=os.path.join(output_root, "spmd_autotune"),
        gemm_tiling_cache_dir=gemm_tiling_cache_dir,
        num_workers=worker_count,
    )
    print(
        f"general_spmd autotune test kernel "
        f"total latency: {spmd_autotune_kernel.latency}s, total energy: {spmd_autotune_kernel.energy}J, "
    )

    spmd_fixed_kernel = general_spmd_fixed_inference(
        cloud_config=cloud_config,
        intermediate_result_dir=os.path.join(output_root, "spmd_fixed"),
        gemm_tiling_cache_dir=gemm_tiling_cache_dir,
        num_workers=worker_count,
    )
    print(
        f"general_spmd fixed test kernel "
        f"total latency: {spmd_fixed_kernel.latency}s, total energy: {spmd_fixed_kernel.energy}J, "
    )

    mpmd_autotune_kernel = general_mpmd_autotune_inference(
        cloud_config=cloud_config,
        intermediate_result_dir=os.path.join(output_root, "mpmd_autotune"),
        gemm_tiling_cache_dir=gemm_tiling_cache_dir,
        num_workers=worker_count,
    )
    print(
        f"general_mpmd autotune test kernel "
        f"total latency: {mpmd_autotune_kernel.latency}s, total energy: {mpmd_autotune_kernel.energy}J, "
    )

    mpmd_fixed_kernel = general_mpmd_fixed_inference(
        cloud_config=cloud_config,
        intermediate_result_dir=os.path.join(output_root, "mpmd_fixed"),
        gemm_tiling_cache_dir=gemm_tiling_cache_dir,
        num_workers=worker_count,
    )
    print(
        f"general_mpmd fixed test kernel "
        f"total latency: {mpmd_fixed_kernel.latency}s, total energy: {mpmd_fixed_kernel.energy}J, "
    )
    return [
        (spmd_autotune_kernel, spmd_autotune_kernel.latency, spmd_autotune_kernel.energy),
        (spmd_fixed_kernel, spmd_fixed_kernel.latency, spmd_fixed_kernel.energy),
        (mpmd_autotune_kernel, mpmd_autotune_kernel.latency, mpmd_autotune_kernel.energy),
        (mpmd_fixed_kernel, mpmd_fixed_kernel.latency, mpmd_fixed_kernel.energy),
    ]


if __name__ == "__main__":
    random.seed(42)

    cloud_chip_config_path = "configs/architecture/system/test_cloud_system.yaml"
    cloud_test_config_list = [
        ("mixtral_8x22b", "configs/models/mixtral_8x22b.json", 2, 8, 16 * 1024),
    ]

    test_cloud_ast_all(
        cloud_chip_config_path,
        cloud_test_config_list,
    )
    edge_chip_config_path = "configs/architecture/system/test_edge_system.yaml"
    edge_test_config_list = [
        ("palm_8b", "configs/models/palm_8b.json", 2, 8, 2 * 1024),
    ]

    test_edge_ast_all(
        edge_chip_config_path,
        edge_test_config_list,
    )
