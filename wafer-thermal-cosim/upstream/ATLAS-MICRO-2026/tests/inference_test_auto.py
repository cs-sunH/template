import csv
import random
from typing import List, Tuple
import os
import sys
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from frontend.hardware_parser import CloudSystemConfig, EdgeSystemConfig
from frontend.model_parser import ModelConfig, get_model_config_from_hf
from frontend.auto.system import CloudSystem, EdgeSystem
from frontend.util import generate_context_slot_mapping


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
            "auto",
            system,
            model_name,
            batch_size,
            context_length,
            num_layers,
            latency,
            energy,
        ])


def test_cloud_inference(
    cloud_config: CloudSystemConfig,
    model_config: ModelConfig,
    context_length_list: List[int],
    element_size: int,
    intermediate_result_dir: str = "",
    gemm_tiling_cache_dir: str = "",
):
    core_num = cloud_config.chip_config.core_num 
    context_slot_mapping = generate_context_slot_mapping(cloud_config, context_length_list)
    if intermediate_result_dir == "":
        intermediate_result_dir = f"kick_the_tires/test_cloud_inference_auto/{model_config.name}_{core_num}cores"
    os.makedirs(intermediate_result_dir, exist_ok=True)

    cloud_system = CloudSystem(
        system_config=cloud_config,
        model_config=model_config,
        element_size=element_size,
    )
    return cloud_system.inference(
        context_length_list=context_length_list,
        context_slot_mapping=context_slot_mapping,
        intermediate_result_dir=intermediate_result_dir,
        gemm_tiling_cache_dir=gemm_tiling_cache_dir,
    )


def test_edge_inference(
    edge_config: EdgeSystemConfig,
    model_config: ModelConfig,
    context_length_list: List[int],
    element_size: int,
    intermediate_result_dir: str = "",
    gemm_tiling_cache_dir: str = "",
    num_tiling_workers: int = -1,
):
    channel_num = edge_config.channel_num
    core_num = edge_config.chip_config.core_num
    if intermediate_result_dir == "":
        intermediate_result_dir = f"kick_the_tires/test_edge_inference_auto/{model_config.name}_{channel_num}ch_{core_num}cores"
    os.makedirs(intermediate_result_dir, exist_ok=True)

    edge_system = EdgeSystem(
        system_config=edge_config,
        model_config=model_config,
        element_size=element_size,
    )
    tiling_kw = {}
    if num_tiling_workers >= 0:
        tiling_kw["num_tiling_workers"] = num_tiling_workers
    return edge_system.inference(
        context_length_list=context_length_list,
        intermediate_result_dir=intermediate_result_dir,
        gemm_tiling_cache_dir=gemm_tiling_cache_dir,
        **tiling_kw,
    )


def test_cloud_all(
    system_config_path: str,
    test_config_list: List[Tuple[str, str, int, int, int]],
    intermediate_result_dir: str = "",
):
    with open(system_config_path, "r") as f:
        cloud_config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    cloud_config = CloudSystemConfig.from_yaml(cloud_config_yaml)
    
    results = []
    for model_name, model_config_path, element_size, batch_size, context_length in test_config_list:
        context_length_list = [context_length] * batch_size
        model_config = get_model_config_from_hf(model_name, model_config_path)
        output_dir = intermediate_result_dir or (
            f"kick_the_tires/test_cloud_inference_auto/"
            f"{model_config.name}_{cloud_config.chip_config.core_num}cores"
        )
        result = test_cloud_inference(
            cloud_config=cloud_config,
            model_config=model_config,
            context_length_list=context_length_list,
            element_size=element_size,
            intermediate_result_dir=output_dir,
        )
        latency = result["latency"]
        energy = result["energy"]
        performance = result["intra_chip_computation_performance"]
        results.append((performance, latency, energy))
        _write_e2e_performance(
            output_dir=output_dir,
            system="cloud",
            model_name=model_name,
            batch_size=batch_size,
            context_length=context_length,
            num_layers=model_config.num_layers,
            latency=latency,
            energy=energy,
        )
        print(
            f"{model_name} batch size {batch_size} context length {context_length} "
            f"total inference latency: {latency}s, total inference energy: {energy}J, "
            f"per layer inference latency: {latency / model_config.num_layers}s, "
            f"per layer inference energy: {energy / model_config.num_layers}J."
        )
    return results


def test_edge_all(
    system_config_path: str,
    test_config_list: List[Tuple[str, str, int, int, int]],
    intermediate_result_dir: str = "",
):
    with open(system_config_path, "r") as f:
        edge_config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    edge_config = EdgeSystemConfig.from_yaml(edge_config_yaml)

    results = []
    for model_name, model_config_path, element_size, batch_size, context_length in test_config_list:
        context_length_list = [context_length] * batch_size
        model_config = get_model_config_from_hf(model_name, model_config_path)
        output_dir = intermediate_result_dir or (
            f"kick_the_tires/test_edge_inference_auto/"
            f"{model_config.name}_{edge_config.channel_num}ch_{edge_config.chip_config.core_num}cores"
        )
        result = test_edge_inference(
            edge_config=edge_config,
            model_config=model_config,
            context_length_list=context_length_list,
            element_size=element_size,
            intermediate_result_dir=output_dir,
        )
        latency = result["latency"]
        energy = result["energy"]
        performance = result["intra_chip_computation_performance"]
        results.append((performance, latency, energy))
        _write_e2e_performance(
            output_dir=output_dir,
            system="edge",
            model_name=model_name,
            batch_size=batch_size,
            context_length=context_length,
            num_layers=model_config.num_layers,
            latency=latency,
            energy=energy,
        )
        print(
            f"{model_name} batch size {batch_size} context length {context_length} "
            f"total inference latency: {latency}s, total inference energy: {energy}J, "
            f"per layer inference latency: {latency / model_config.num_layers}s, "
            f"per layer inference energy: {energy / model_config.num_layers}J."
        )
    return results


if __name__ == "__main__":
    random.seed(42)

    cloud_chip_config_path = "configs/architecture/system/test_cloud_system.yaml"
    cloud_test_config_list = [
        ("opt_66b", "configs/models/opt_66b.json", 2, 8, 4*1024),
    ]

    test_cloud_all(
        cloud_chip_config_path,
        cloud_test_config_list,
    )

    edge_chip_config_path = "configs/architecture/system/test_edge_system.yaml"
    edge_test_config_list = [
        ("opt_6.7b", "configs/models/opt_6.7b.json", 2, 8, 2*1024),
    ]
    test_edge_all(
        edge_chip_config_path,
        edge_test_config_list,
    )
