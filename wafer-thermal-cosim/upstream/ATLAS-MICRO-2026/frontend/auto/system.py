import os
import yaml
import math
from typing import List, Dict, Any, Tuple
import multiprocessing
import copy
from collections import defaultdict

from atlasim import Chip, SimulatorOperatorType

from frontend.model_parser import Operator, OperatorType, ModelConfig, ParallelConfig, get_layer_operator_list
from frontend.hardware_parser import CloudSystemConfig, EdgeSystemConfig
from frontend.auto.tiling_explorer_cloud import explore_cloud_tiling
from frontend.auto.tiling_explorer_edge import explore_edge_tiling
from frontend.util import _MHz


def _build_dattn_input_payload(
    attention_input_template: Dict[str, Any],
    context_slot_mapping: List[Dict[int, List[int]]],
    core_num: int,
) -> Dict[str, Any]:
    token_tile_size = int(attention_input_template["token_tile_size"])
    input_data = {
        # Attention shape information
        "q_head_num": attention_input_template["q_head_num"],
        "kv_head_num": attention_input_template["kv_head_num"],
        "q_head_num_per_kv": attention_input_template["q_head_num_per_kv"],
        "qk_head_dim": attention_input_template["qk_head_dim"],
        "v_head_dim": attention_input_template["v_head_dim"],

        # Tensor information
        "input_tensor_name": attention_input_template["input_tensor_name"],
        "output_tensor_name": attention_input_template["output_tensor_name"],
        "kv_cache_tensor_name": attention_input_template["kv_cache_tensor_name"],
        "total_slot_num": attention_input_template["total_slot_num"],
        "kv_head_addr_offset": attention_input_template["kv_head_addr_offset"],

        # Token number information for decoding attention
        "token_tile_size": token_tile_size,
        "block_size": attention_input_template["block_size"],

        # Whether randomly assign token slots to each request or not
        "random": attention_input_template.get("random", False),
        "random_seed": attention_input_template.get("random_seed", 0),

        "core_input_list": []
    }

    for core_id in range(core_num):
        core_mapping = context_slot_mapping[core_id]
        request_indices = list(core_mapping.keys())
        request_tile_count = []
        request_tile_info_list = []
        for request_index in request_indices:
            slot_id_list = core_mapping[request_index]
            request_tile_count.append(math.ceil(len(slot_id_list) / token_tile_size))

            request_tile_info = []
            for tile_start in range(0, len(slot_id_list), token_tile_size):
                request_tile_info.append(slot_id_list[tile_start : tile_start + token_tile_size])
            request_tile_info_list.append(request_tile_info)

        input_data["core_input_list"].append(
            {
                "request_indices": request_indices,
                "request_tile_count": request_tile_count,
                "request_tile_info_list": request_tile_info_list,
            }
        )

    return input_data


class CloudSystem:
    def __init__(
        self,
        # Hardware config
        system_config: CloudSystemConfig,
        # Software config
        model_config: ModelConfig,
        element_size: int,
    ):
        self.system_config = system_config
        self.parallel_config = system_config.parallel_config

        self.model_config = model_config
        self.element_size = element_size
        
        self.online_initialized = False
        self.instance = None

    def estimate_inter_chip_communication_performance(
        self,
        inter_chip_communication_list: List[Operator],
        batch_size: int = -1,
    ):
        performance_dict: Dict[str, Tuple[float, float]] = {}

        if self.system_config.tp_scope == "scale_up":
            tp_bandwidth = self.system_config.interconnect_config.scale_up_bandwidth
            tp_latency = self.system_config.interconnect_config.scale_up_latency
            tp_energy = self.system_config.interconnect_config.scale_up_energy
        elif self.system_config.tp_scope == "scale_out":
            tp_bandwidth = self.system_config.interconnect_config.scale_out_bandwidth
            tp_latency = self.system_config.interconnect_config.scale_out_latency
            tp_energy = self.system_config.interconnect_config.scale_out_energy
        else:
            assert False, "only scale_up and scale_out are supported for tp_scope"

        if self.system_config.ep_scope == "scale_up":
            ep_bandwidth = self.system_config.interconnect_config.scale_up_bandwidth
            ep_latency = self.system_config.interconnect_config.scale_up_latency
            ep_energy = self.system_config.interconnect_config.scale_up_energy
        elif self.system_config.ep_scope == "scale_out":
            ep_bandwidth = self.system_config.interconnect_config.scale_out_bandwidth
            ep_latency = self.system_config.interconnect_config.scale_out_latency
            ep_energy = self.system_config.interconnect_config.scale_out_energy
        else:
            assert False, "only scale_up and scale_out are supported for ep_scope"

        for operator in inter_chip_communication_list:
            if batch_size > 0:
                operator.M = batch_size

            if operator.op_type == OperatorType.ALLREDUCE:
                per_rank_data_colume = operator.get_memory_capacity(self.element_size)
                total_ar_volume = 2 * per_rank_data_colume * (self.parallel_config.tp_size-1) / self.parallel_config.tp_size # Byte
                
                latency = max(
                    total_ar_volume / (tp_bandwidth * (2 ** 30)),
                    tp_latency
                )
                energy = total_ar_volume * 8 * tp_energy / (10 ** 12) # from pJ to J
                performance_dict[operator.name] = (latency, energy)
            elif operator.op_type == OperatorType.ALL2ALL:
                total_data_volume = operator.get_memory_capacity(self.element_size) * self.parallel_config.ep_size # Byte
                
                latency = max(
                    total_data_volume / (ep_bandwidth * (2 ** 30)),
                    ep_latency
                )
                energy = total_data_volume * 8 * ep_energy / (10 ** 12) # from pJ to J
                performance_dict[operator.name] = (latency, energy)
            else:
                assert False, "only ALLREDUCE and ALL2ALL are supported for inter-chip communication"
       
        return performance_dict

    def inference(
        self,
        # Intra-chip KV cache description
        context_length_list: List[int],
        context_slot_mapping: List[Dict[int, List[int]]],  # {core id : {request intra-batch id : token slot id list}}
        # Intermediate result directory
        intermediate_result_dir: str = "",
        # Shared GEMM tiling cache directory (reusable across context lengths)
        gemm_tiling_cache_dir: str = "",
    ):
        batch_size = len(context_length_list)
        if intermediate_result_dir == "":
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            intermediate_result_dir = os.path.join(
                project_root,
                (
                    "decaparated/cloud_inference/"
                    f"{self.model_config.name}_"
                    f"tp{self.parallel_config.tp_size}_"
                    f"ep{self.parallel_config.ep_size}_"
                    f"bs{batch_size}"
                )
            )
        os.makedirs(intermediate_result_dir, exist_ok=True)

        core_num = self.system_config.chip_config.core_num
        core_flit_index_list = [0 for _ in range(core_num)]
        attention_block, ffn_moe_block = get_layer_operator_list(
            model_config=self.model_config,
            parallel_config=self.parallel_config
        )

        cloud_tiling_result = explore_cloud_tiling(
            # Operator-related configs
            model_config=self.model_config,
            attention_block=attention_block,
            ffn_moe_block=ffn_moe_block,
            # Input-related configs
            context_length_list=context_length_list,
            core_flit_index_list=core_flit_index_list,
            # Hardware-related configs
            cloud_config=self.system_config,
            # Some hyper parameters
            element_size=self.element_size,
            intermediate_result_dir=f"{intermediate_result_dir}/operator_tling_exploration",
            gemm_tiling_cache_dir=gemm_tiling_cache_dir,
        )
        data_placement_list = cloud_tiling_result["data_placement_list"]
        task_description_list = cloud_tiling_result["task_description_list"]
        attention_input_template = cloud_tiling_result["attention_input_template"]
        inter_chip_communication_list = cloud_tiling_result["inter_chip_communication_list"]
        for task_description in task_description_list:
            if task_description[0] == "computation" and len(task_description) == 4:
                print(f"computation {task_description[1]['name']} latency: {task_description[3]}")
            elif task_description[0] == "communication" and len(task_description) == 3:
                print(f"communication {[x[0]['name'] for x in task_description[1]]} latency: {task_description[2]}")

        # Store data placement config
        data_placement_config_path = os.path.join(intermediate_result_dir, "data_placement.yaml")
        with open(data_placement_config_path, "w") as f:
            yaml.dump({"tensor" : data_placement_list}, f)

        has_decode_attention = any(
            task_description[0] == "computation"
            and task_description[1].get("type") == str(SimulatorOperatorType.DecodeAttention)
            for task_description in task_description_list
        )
        attention_input_config_path = ""
        if has_decode_attention:
            if attention_input_template is None:
                raise ValueError("Cloud decode-attention operator requires an attention input template.")
            attention_input_config_path = os.path.abspath(os.path.join(intermediate_result_dir, "attn_input.yaml"))
            attention_input_payload = _build_dattn_input_payload(
                attention_input_template=attention_input_template,
                context_slot_mapping=context_slot_mapping,
                core_num=self.system_config.chip_config.core_num,
            )
            with open(attention_input_config_path, "w") as f:
                yaml.dump({"attention_input": attention_input_payload}, f, sort_keys=False)

        # Store operator description config
        operator_description_config_path = os.path.join(intermediate_result_dir, "operator_description.yaml")
        operator_description = {"operator" : []}
        file_prefix_totals: Dict[str, int] = defaultdict(int)
        for task_description in task_description_list:
            if task_description[0] != "communication":
                continue
            communication_descripton_list: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = task_description[1]
            for communication_description in communication_descripton_list:
                file_prefix_totals[str(communication_description[0]["file_prefix"]).rstrip("/")] += 1
        file_prefix_indices: Dict[str, int] = defaultdict(int)
        for task_description in task_description_list:
            if task_description[0] == "computation":
                op_description = copy.deepcopy(task_description[1])
                if op_description.get("type") == str(SimulatorOperatorType.DecodeAttention):
                    op_description["attention_input"] = attention_input_config_path
                operator_description["operator"].append(op_description)
            elif task_description[0] == "communication":
                communication_descripton_list: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = task_description[1]
                for communication_description in communication_descripton_list:
                    op_description = copy.deepcopy(communication_description[0])
                    per_core_description_list = communication_description[1]

                    raw_file_prefix = str(op_description["file_prefix"]).rstrip("/")
                    file_prefix_index = file_prefix_indices[raw_file_prefix]
                    file_prefix_indices[raw_file_prefix] += 1
                    if file_prefix_totals[raw_file_prefix] > 1:
                        unique_file_prefix = f"{raw_file_prefix}/instance{file_prefix_index}"
                    else:
                        unique_file_prefix = raw_file_prefix

                    op_description["file_prefix"] = os.path.join(intermediate_result_dir, unique_file_prefix)
                    os.makedirs(op_description["file_prefix"], exist_ok=True)
                    operator_description["operator"].append(op_description)
                    
                    # For communication operator, further store per-core execution description
                    for i, per_core_description in enumerate(per_core_description_list):
                        with open(os.path.join(op_description["file_prefix"], f"core_{i}.yaml"), "w") as f:
                            yaml.dump(per_core_description, f)
        with open(operator_description_config_path, "w") as f:
            yaml.dump(operator_description, f)
        
        # Create Chip object, and conduct simulation
        chip = Chip(
            arch_config_path=self.system_config.chip_config_path,
            operator_list_path=operator_description_config_path,
            placement_map_path=data_placement_config_path,
        )
        intra_chip_computation_performance = chip.simulate()
        del chip
        intra_chip_computation_latency = intra_chip_computation_performance.e2e_stats.e2e_cycles / (self.system_config.chip_config.frequency * _MHz)

        # Estimate inter-chip communication performance
        inter_chip_communication_performance = self.estimate_inter_chip_communication_performance(
            inter_chip_communication_list=inter_chip_communication_list,
        )
        inter_chip_communication_latency = 0
        inter_chip_communication_energy = 0
        for performance_name, performance_value in inter_chip_communication_performance.items():
            inter_chip_communication_latency += performance_value[0]
            inter_chip_communication_energy += performance_value[1]

        chip_config = self.system_config.chip_config
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

        total_latency = (intra_chip_computation_latency + inter_chip_communication_latency) * self.model_config.num_layers
        energy = (
            intra_chip_computation_performance.e2e_stats.e2e_energy
            + inter_chip_communication_energy
            + inter_chip_communication_latency * chip_active_power
        ) * self.model_config.num_layers

        return {
            "intra_chip_computation_performance": intra_chip_computation_performance,
            "intra_chip_computation_latency": intra_chip_computation_latency,
            "intra_chip_computation_energy": intra_chip_computation_performance.e2e_stats.e2e_energy,
            "inter_chip_communication_performance": inter_chip_communication_performance,
            "inter_chip_communication_latency": inter_chip_communication_latency,
            "inter_chip_communication_energy": inter_chip_communication_energy + inter_chip_communication_latency * chip_active_power,
            "latency": total_latency,
            "energy": energy,
        }


class EdgeSystem:
    def __init__(
        self,
        # Hardware config
        system_config: EdgeSystemConfig,
        # Software config
        model_config: ModelConfig,
        element_size: int,
    ) -> None:
        self.system_config = system_config
        self.parallel_config = ParallelConfig(
            tp_size=1,
            ep_size=1,
        )
        self.model_config = model_config
        self.element_size = element_size
        self.instance = None
    
    def estimate_softmax_performance(
        self,
        inter_chip_communication_list: List[Operator],
        task_description_list: List[Tuple[str, Dict[str, Any]]],
        context_length_list: List[int],
    ):
        if not inter_chip_communication_list:
            return {
                "softmax_latency": 0.0,
                "softmax_energy": 0.0,
            }
        assert inter_chip_communication_list[0].name == "softmax"

        # Find attention_qk task description for softmax estimation
        attention_qk_desc = None
        for td in task_description_list:
            if td[0] == "gemm" and td[1]["name"] in ("attention_qk", "mla_attention_qk"):
                attention_qk_desc = td[1]
                break
        assert attention_qk_desc is not None, "attention_qk operator not found in task_description_list"

        intra_vec_len = self.system_config.intra_channel_vector_length

        softmax_latency = float('inf')
        softmax_energy = 0

        batch_size = len(context_length_list)
        context_length = context_length_list[0]
        kv_group_num = self.model_config.n_kv_group
        n_kv_head = self.model_config.n_kv_head

        npu_softmax_op_count = 9
        npu_vec_count = npu_softmax_op_count * n_kv_head * batch_size * kv_group_num * context_length
        npu_latency = npu_vec_count / (self.system_config.npu_vec_length * self.system_config.npu_frequency * _MHz)
        npu_energy = npu_vec_count * self.system_config.npu_vec_energy * 1e-12
        if npu_latency < softmax_latency:
            softmax_latency = npu_latency
            softmax_energy = npu_energy

        if intra_vec_len > 0:
            channel_B = attention_qk_desc["gemm_num_per_channel"]
            M = attention_qk_desc["output_placement"]["shape"][0]
            core_N = attention_qk_desc["output_placement"]["shape"][1]
            core_nN = attention_qk_desc["intra_channel_tiling_factors"][1]
            channel_N = core_N * core_nN
            per_channel_elements = channel_B * M * channel_N
            vc = per_channel_elements * 9
            latency = vc / (intra_vec_len * self.system_config.chip_config.frequency * _MHz)
            if latency < softmax_latency:
                softmax_latency = latency
                softmax_energy = latency * self.system_config.chip_config.vector_config.power
        
        return {
            "softmax_latency": softmax_latency,
            "softmax_energy": softmax_energy,
        }

    def inference(
        self,
        # Intra-chip KV cache description
        context_length_list: List[int],
        # Intermediate result directory
        intermediate_result_dir: str = "",
        # Shared GEMM tiling cache directory (reusable across context lengths)
        gemm_tiling_cache_dir: str = "",
        # Number of workers for tiling exploration (0 = sequential, useful to
        # avoid nested-fork deadlocks when already inside a child process)
        num_tiling_workers: int = int(0.8 * multiprocessing.cpu_count()),
    ):
        batch_size = len(context_length_list)
        if intermediate_result_dir == "":
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            intermediate_result_dir = os.path.join(
                project_root,
                (
                    "decaparated/edge_inference/"
                    f"{self.model_config.name}_"
                    f"tp{self.parallel_config.tp_size}_"
                    f"ep{self.parallel_config.ep_size}_"
                    f"bs{batch_size}"
                )
            )
        os.makedirs(intermediate_result_dir, exist_ok=True)

        attention_block, ffn_moe_block = get_layer_operator_list(
            model_config=self.model_config,
            parallel_config=self.parallel_config,
            skip_communication=True,
            non_fused_attention=True,
        )

        edge_tiling_result = explore_edge_tiling(
            # Operator-related configs
            model_config=self.model_config,
            attention_block=attention_block,
            ffn_moe_block=ffn_moe_block,
            # Input-related configs
            context_length_list=context_length_list,
            # Hardware-related configs
            edge_config=self.system_config,
            # Some hyper parameters
            element_size=self.element_size,
            num_workers=num_tiling_workers,
            intermediate_result_dir=f"{intermediate_result_dir}/operator_tling_exploration",
            gemm_tiling_cache_dir=gemm_tiling_cache_dir,
        )
        data_placement_list = edge_tiling_result["data_placement_list"]
        task_description_list = edge_tiling_result["task_description_list"]
        inter_chip_communication_list = edge_tiling_result["inter_chip_communication_list"]
        for task_description in task_description_list:
            if task_description[0] == "gemm":
                print(
                    f"gemm {task_description[1]['name']} latency: {task_description[1]['latency']}, "
                    f"computation latency: {task_description[1]['computation_latency']}, "
                    f"intra channel accumulation latency: {task_description[1]['intra_channel_accumulation_latency']}, "
                    f"inter channel communication latency: {task_description[1]['inter_channel_communication_latency']}."
                )
            elif task_description[0] == "attention":
                print(
                    f"attention {task_description[1]['name']} latency: {task_description[1]['latency']}, "
                    f"inter channel communication latency: {task_description[1]['inter_channel_communication_latency']}."
                )
        
        # Store data placement config
        data_placement_config_path = os.path.join(intermediate_result_dir, "data_placement.yaml")
        with open(data_placement_config_path, "w") as f:
            yaml.dump({"tensor" : data_placement_list}, f)
        
        operator_description_path = os.path.join(intermediate_result_dir, "operator_description.yaml")
        operator_description = {"operator" : []}
        inter_channel_communication_total_latency = 0
        intra_channel_accumulation_total_latency = 0
        inter_channel_communication_total_energy = 0
        intra_channel_accumulation_total_energy = 0
        intra_channel_accumulation_latency_dict: Dict[str, float] = {}
        intra_channel_accumulation_energy_dict: Dict[str, float] = {}
        inter_channel_communication_latency_dict: Dict[str, float] = {}
        inter_channel_communication_energy_dict: Dict[str, float] = {}
        for task_description in task_description_list:
            if task_description[0] == "gemm":
                operator_description["operator"].extend(task_description[1]["description"])

                inter_channel_communication_total_latency += task_description[1]["inter_channel_communication_latency"]
                inter_channel_communication_total_energy += task_description[1].get("inter_channel_communication_energy", 0)
                intra_channel_accumulation_total_latency += task_description[1]["intra_channel_accumulation_latency"]
                intra_channel_accumulation_total_energy += task_description[1].get("intra_channel_accumulation_energy", 0)
                op_name = task_description[1]["name"]
                intra_channel_accumulation_latency_dict[op_name] = task_description[1]["intra_channel_accumulation_latency"]
                intra_channel_accumulation_energy_dict[op_name] = task_description[1].get("intra_channel_accumulation_energy", 0)
                inter_channel_communication_latency_dict[op_name] = task_description[1]["inter_channel_communication_latency"]
                inter_channel_communication_energy_dict[op_name] = task_description[1].get("inter_channel_communication_energy", 0)
        with open(operator_description_path, "w") as f:
            yaml.dump(operator_description, f)
        
        # Create Chip object
        chip = Chip(
            arch_config_path=self.system_config.chip_config_path,
            operator_list_path=operator_description_path,
            placement_map_path=data_placement_config_path,
        )
        
        performance = chip.simulate()
        latency = performance.e2e_stats.e2e_cycles / (self.system_config.chip_config.frequency * _MHz)

        softmax_performance = self.estimate_softmax_performance(
            inter_chip_communication_list=inter_chip_communication_list,
            task_description_list=task_description_list,
            context_length_list=context_length_list,
        )
        softmax_latency = softmax_performance["softmax_latency"]
        softmax_energy = softmax_performance["softmax_energy"]

        chip_total_energy = performance.e2e_stats.e2e_energy
        total_latency = (
            latency
            + intra_channel_accumulation_total_latency
            + inter_channel_communication_total_latency
            + softmax_latency
        ) * self.model_config.num_layers
        energy = (
            chip_total_energy
            + intra_channel_accumulation_total_energy
            + inter_channel_communication_total_energy
            + softmax_energy
        ) * self.model_config.num_layers
        
        del chip
        return {
            "intra_chip_computation_performance": performance,
            "intra_chip_computation_latency": latency,
            "intra_chip_computation_energy": performance.e2e_stats.e2e_energy,
            "intra_channel_accumulation_latency_dict": intra_channel_accumulation_latency_dict,
            "intra_channel_accumulation_total_latency": intra_channel_accumulation_total_latency,
            "intra_channel_accumulation_energy_dict": intra_channel_accumulation_energy_dict,
            "intra_channel_accumulation_total_energy": intra_channel_accumulation_total_energy,
            "inter_channel_communication_latency_dict": inter_channel_communication_latency_dict,
            "inter_channel_communication_total_latency": inter_channel_communication_total_latency,
            "inter_channel_communication_energy_dict": inter_channel_communication_energy_dict,
            "inter_channel_communication_total_energy": inter_channel_communication_total_energy,
            "softmax_latency": softmax_latency,
            "softmax_energy": softmax_energy,
            "latency": total_latency,
            "energy": energy,
        }
