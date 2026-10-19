import yaml
import math
import os
import contextlib
import multiprocessing
from typing import List, Dict, Any, Tuple
from itertools import product
import copy

from atlasim import Chip, SimulatorOperatorType

from frontend.model_parser import Operator, OperatorType, ModelConfig
from frontend.hardware_parser import EdgeSystemConfig
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


def set_shape(
    model_config: ModelConfig,
    attention_block: List[Operator],
    ffn_moe_block: List[Operator],
    context_length_list: List[int],
):
    batch_size = len(context_length_list)
    for operator in attention_block:
        if operator.name == "softmax":
            continue
        if operator.op_type == OperatorType.GEMM:
            if operator.name in ("attention_qk", "mla_attention_qk"):
                assert all(ctx == context_length_list[0] for ctx in context_length_list), \
                    "All context lengths must be the same for non-fused attention in edge inference."
                operator.B *= batch_size
                operator.N = context_length_list[0]
            elif operator.name in ("attention_sv", "mla_attention_sv"):
                operator.B *= batch_size
                operator.K = context_length_list[0]
            else:
                operator.M = batch_size
        elif operator.op_type == OperatorType.ATTENTION:
            operator.input_length = [1 for _ in range(len(context_length_list))]
            operator.context_length = [c for c in context_length_list]
    
    if model_config.is_moe:
        # assume a uniform distribution of expert routing
        per_ffn_token_num = math.ceil(batch_size * (model_config.moe_top_k + model_config.moe_shared_experts) / model_config.moe_num_experts)
        total_ffn_num = math.ceil(min(batch_size * (model_config.moe_top_k + model_config.moe_shared_experts), model_config.moe_num_experts))
    else:
        total_ffn_num = 1
        per_ffn_token_num = batch_size
    for operator in ffn_moe_block:
        if operator.op_type == OperatorType.GEMM:
            operator.B = total_ffn_num
            operator.M = per_ffn_token_num


def _generate_gemm_data_placements(operator, element_size, dram_row_size, base_addr):
    cur_base_addr = base_addr
    input_placement = make_tensor_placement(
        name="input_" + operator.name,
        base_addr=cur_base_addr,
        element_size=element_size,
        shape=[operator.M, operator.K],
        strides=make_contiguous_strides([operator.M, operator.K], last_dim_contiguous=True),
    )
    cur_base_addr += math.ceil(operator.M * operator.K * element_size / dram_row_size) * dram_row_size

    output_placement = make_tensor_placement(
        name="output_" + operator.name,
        base_addr=cur_base_addr,
        element_size=element_size,
        shape=[operator.M, operator.N],
        strides=make_contiguous_strides([operator.M, operator.N], last_dim_contiguous=True),
    )
    cur_base_addr += math.ceil(operator.M * operator.N * element_size / dram_row_size) * dram_row_size

    weight_placement_list = []
    for i in range(operator.B):
        weight_placement = make_tensor_placement(
            name="weight_" + operator.name + f"_b{i}",
            base_addr=cur_base_addr,
            element_size=element_size,
            shape=[operator.K, operator.N],
            strides=make_contiguous_strides([operator.K, operator.N], last_dim_contiguous=False),
        )
        cur_base_addr += math.ceil(operator.K * operator.N * element_size / dram_row_size) * dram_row_size
        weight_placement_list.append(weight_placement)

    return {
        "input_placement": input_placement,
        "output_placement": output_placement,
        "weight_placement_list": weight_placement_list,
        "next_base_addr": cur_base_addr,
    }


def explore_gemm_comp_worker(
    # Tensor description
    input_placement: Dict[str, Any],
    output_placement: Dict[str, Any],
    weight_placement_list: List[Dict[str, Any]],
    # Operator shape description
    operator: Operator,
    per_output_accumulate_count: int,
    element_size: int,
    # Edge system config (picklable subset)
    edge_config_dict: Dict[str, Any],
    # Tiling factors to evaluate
    tiling_factor_combination_list: List[Tuple[int, int, int]],
    # Hyper parameters
    worker_id: int,
    cur_op_dir: str,
    candidate_id: int = 0,
):
    cur_worker_dir = os.path.join(cur_op_dir, f"worker_{worker_id}")
    os.makedirs(cur_worker_dir, exist_ok=True)

    data_placement_config = {
        "tensor": [
            input_placement,
            output_placement,
            *weight_placement_list,
        ]
    }
    data_placement_config_path = os.path.join(cur_worker_dir, "data_placement.yaml")
    with open(data_placement_config_path, "w") as f:
        yaml.dump(data_placement_config, f)
    # Only use the first weight placement for tiling exploration
    input_name = input_placement["name"]
    output_name = output_placement["name"]
    weight_name = weight_placement_list[0]["name"]

    opt_comp_description = {
        "description": None,
        "tiling_factors": (-1, -1, -1),
        "computation_latency": float('inf'),
        "intra_channel_accumulation_latency": float('inf'),
    }
    opt_latency = float('inf')
    comp_description_path = os.path.join(cur_worker_dir, "comp_description.yaml")
    for tiling_factor_combination in tiling_factor_combination_list:
        tM, tK, tN = tiling_factor_combination
        nM = math.ceil(operator.M / tM)
        nK = math.ceil(operator.K / tK)
        nN = math.ceil(operator.N / tN)
        mac_count = tM*tN*tK
        mac_buffer_load_count = element_size * (tM*tK + tK*tN)
        mac_buffer_store_count = element_size * (tM*tN)
        vec_count = tM*tN
        vec_buffer_load_count = element_size * (tM*tN)
        vec_buffer_store_count = element_size * (tM*tN)

        comp_description = {
            "operator": [{
                "name": operator.name,
                "type": str(SimulatorOperatorType.GEMM),
                "iteration": nM*nK*nN,
                "execution": {
                    "matrix": [
                        {
                            "name": "gemm_tile",
                            "mac_count": mac_count,
                        }
                    ],
                    "vector": [
                        {
                            "name": "gemm_tile_accumulation",
                            "vec_count": vec_count,
                        }
                    ],
                    "buffer_load": [
                        {
                            "name": "gemm_tile_load",
                            "byte_count": mac_buffer_load_count,
                            "is_write": False,
                        },
                        {
                            "name": "gemm_tile_accumulation_load",
                            "byte_count": vec_buffer_load_count,
                            "is_write": False,
                        }
                    ],
                    "buffer_store": [
                        {
                            "name": "gemm_tile_store",
                            "byte_count": mac_buffer_store_count,
                            "is_write": True,
                        },
                        {
                            "name": "gemm_tile_accumulation_store",
                            "byte_count": vec_buffer_store_count,
                            "is_write": True,
                        }
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
                            # In output stationary, when nK=1, all nN tiles in the same nM rows
                            # can share the same input tile.
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
                            # In output stationary, when nK*nN=1, we only need to load the full weight for one time
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
                    ]
                }
            }]
        }
        with open(comp_description_path, "w") as f:
            yaml.dump(comp_description, f)
        
        # Redirect C++ output to a log file
        import atlasim
        log_file_path = os.path.join(cur_worker_dir, "worker_output.log")
        with open(log_file_path, "w") as f:
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                with atlasim.ostream_redirect(stdout=True, stderr=True):
                    chip = Chip(edge_config_dict["chip_config_path"], comp_description_path, data_placement_config_path)
                    performance = chip.simulate()
                    del chip
        computation_latency = performance.e2e_stats.e2e_cycles / (edge_config_dict["frequency"] * _MHz)

        # Estimate intra-channel accumulation latency and energy
        intra_channel_accumulation_latency = 0
        intra_channel_accumulation_energy = 0
        if per_output_accumulate_count > 0 and edge_config_dict["intra_channel_vector_length"] > 0:
            per_core_output_element_num = operator.M * operator.N
            per_channel_vec_op_count = per_core_output_element_num * per_output_accumulate_count
            accumulation_latency = per_channel_vec_op_count / (edge_config_dict["intra_channel_vector_length"] * edge_config_dict["frequency"] * _MHz)
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
        "computation_latency": float('inf'),
        "intra_channel_accumulation_latency": float('inf'),
        "intra_channel_accumulation_energy": 0,
        "latency": float('inf'),
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
        all_batch_comp_description["latency"] = opt_comp_description["computation_latency"] + opt_comp_description["intra_channel_accumulation_latency"]
    
    all_batch_comp_description["candidate_id"] = candidate_id
    return all_batch_comp_description


def explore_gemm_comp(
    # Tensor description
    base_addr: int,
    # Operator shape description
    operator: Operator,
    element_size: int,
    # Edge system config
    edge_config: EdgeSystemConfig,
    # Hyper parameters
    dram_row_size: int,
    min_tM: int, min_tK: int, min_tN: int,
    num_workers: int,
    # Intermediate result directory
    intermediate_result_dir: str,
    # Shared GEMM tiling cache directory (reusable across context lengths)
    gemm_tiling_cache_dir: str = "",
):
    assert operator.op_type == OperatorType.GEMM
    cache_base_dir = gemm_tiling_cache_dir if gemm_tiling_cache_dir else intermediate_result_dir
    cur_op_dir = os.path.join(cache_base_dir, operator.name)
    os.makedirs(cur_op_dir, exist_ok=True)

    cache_file_name = f"gemm_opt_B{operator.B}_M{operator.M}_K{operator.K}_N{operator.N}_general_layout.yaml"
    cache_file_path = os.path.join(cur_op_dir, cache_file_name)
    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as f:
            cached = yaml.safe_load(f)
        if (cached is not None
            and cached.get("M") == operator.M
            and cached.get("K") == operator.K
            and cached.get("N") == operator.N
            and cached.get("B") == operator.B
            and cached.get("element_size") == element_size
            and cached.get("core_M") is not None):
            print(f"[Cache hit] {operator.name}: core tiling {cached['core_tiling_factors']}, "
                  f"intra {cached['intra_channel_tiling_factors']}, "
                  f"inter {cached['inter_channel_tiling_factors']}")
            core_op = copy.deepcopy(operator)
            core_op.M = cached["core_M"]
            core_op.K = cached["core_K"]
            core_op.N = cached["core_N"]
            core_op.B = cached["channel_B"]
            placement_result = _generate_gemm_data_placements(core_op, element_size, dram_row_size, base_addr)
            input_placement = placement_result["input_placement"]
            output_placement = placement_result["output_placement"]
            weight_placement_list = placement_result["weight_placement_list"]
            next_base_addr = placement_result["next_base_addr"]
            return {
                "name": operator.name,
                "description": cached["description"],
                "gemm_num_per_channel": cached["gemm_num_per_channel"],
                "core_tiling_factors": tuple(cached["core_tiling_factors"]),
                "intra_channel_tiling_factors": tuple(cached["intra_channel_tiling_factors"]),
                "inter_channel_tiling_factors": tuple(cached["inter_channel_tiling_factors"]),
                "channel_group_num": cached["channel_group_num"],
                "computation_latency": cached["computation_latency"],
                "intra_channel_accumulation_latency": cached["intra_channel_accumulation_latency"],
                "intra_channel_accumulation_energy": cached["intra_channel_accumulation_energy"],
                "inter_channel_communication_latency": cached["inter_channel_communication_latency"],
                "inter_channel_communication_energy": cached["inter_channel_communication_energy"],
                "latency": cached["latency"],
                "next_base_addr": next_base_addr,
                "input_placement": input_placement,
                "output_placement": output_placement,
                "weight_placement_list": weight_placement_list,
            }

    edge_config_dict = {
        "chip_config_path": edge_config.chip_config_path,
        "frequency": edge_config.chip_config.frequency,
        "core_num": edge_config.chip_config.core_num,
        "intra_channel_vector_length": edge_config.intra_channel_vector_length,
        "vector_power": edge_config.chip_config.vector_config.power,
    }
    buffer_size = edge_config.chip_config.buffer_config.buffer_size * _KB

    channel_num = edge_config.channel_num
    channel_num_factor_list = get_factors(channel_num)
    core_num = edge_config.chip_config.core_num
    core_num_factor_list = get_factors(core_num) if edge_config.intra_channel_vector_length > 0 else [1]

    # Phase 1: Enumerate all (channel_factor, channel_nK, core_nK) candidates
    candidates = []
    for channel_num_factor in channel_num_factor_list:
        # First split GEMMs and channels into channel_num_factor groups.
        # In each group, all GEMMs adopt the same tiling & execution flow across the corresponding channels.
        if channel_num_factor > operator.B:
            continue
        B_per_group = math.ceil(operator.B / channel_num_factor)
        channel_per_group = math.ceil(channel_num / channel_num_factor)

        channel_per_group_factor_list = get_factors(channel_per_group)
        for channel_nK in channel_per_group_factor_list:
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

                can_intra_accum = (core_nK > 1 and edge_config.intra_channel_vector_length > 0)
                intra_accum_opts = [True, False] if can_intra_accum else [False]

                for use_intra_accum in intra_accum_opts:
                    per_output_accumulate_count = (core_nK - 1) if use_intra_accum else 0

                    input_scatter_data_volume = channel_B * channel_M * channel_K * element_size
                    output_gather_data_volume = channel_B * channel_M * channel_N * element_size

                    if not use_intra_accum and core_nK > 1:
                        output_gather_data_volume *= core_nK

                    inter_channel_communication_latency = (input_scatter_data_volume + output_gather_data_volume) / (edge_config.channel_bandwidth * _GB)
                    inter_channel_communication_energy = (input_scatter_data_volume + output_gather_data_volume) * 8 * edge_config.channel_energy * 1e-12

                    if channel_nK > 1 and edge_config.npu_vec_length > 0:
                        per_output_accumulate_count_npu = channel_nK - 1
                        npu_vec_op_count = channel_B * channel_M * channel_N * per_output_accumulate_count_npu
                        npu_vec_latency = npu_vec_op_count / (edge_config.npu_vec_length * edge_config.npu_frequency * _MHz)
                        npu_vec_energy = npu_vec_op_count * edge_config.npu_vec_energy * 1e-12
                        inter_channel_communication_latency = max(npu_vec_latency, inter_channel_communication_latency)
                        inter_channel_communication_energy += npu_vec_energy

                    # Generate all possible tiling factor combinations
                    cur_min_tM, cur_min_tK, cur_min_tN = min_tM, min_tK, min_tN
                    tM_list = get_factors(core_M, cur_min_tM)
                    tK_list = get_factors(core_K, cur_min_tK)
                    tN_list = get_factors(core_N, cur_min_tN)
                    while len(tM_list) == 0:
                        tM_list = [core_M]
                    while len(tK_list) == 0:
                        cur_min_tK = cur_min_tK // 2
                        tK_list = get_factors(core_K, cur_min_tK)
                    while len(tN_list) == 0:
                        cur_min_tN = cur_min_tN // 2
                        tN_list = get_factors(core_N, cur_min_tN)
                    if len(tK_list) == 0:
                        tK_list = get_factors(core_K)
                    if len(tN_list) == 0:
                        tN_list = get_factors(core_N)
                    all_tilings = list(product(tM_list, tK_list, tN_list))
                    legal_tilings = [
                        t for t in all_tilings
                        if element_size * (t[0]*t[1] + t[1]*t[2] + t[0]*t[2]) <= buffer_size / 2
                    ]
                    if not legal_tilings:
                        continue

                    core_placement_result = _generate_gemm_data_placements(core_operator, element_size, dram_row_size, 0)
                    core_input_placement = core_placement_result["input_placement"]
                    core_output_placement = core_placement_result["output_placement"]
                    core_weight_placement_list = core_placement_result["weight_placement_list"]

                    candidates.append({
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
                    })

    if not candidates:
        return None

    # Phase 2: Distribute work across workers proportionally by tiling count
    work_items = []
    total_tilings = sum(len(c["legal_tilings"]) for c in candidates)
    for cand_id, cand in enumerate(candidates):
        workers_for_cand = max(1, round(len(cand["legal_tilings"]) / total_tilings * num_workers))
        partitions = partition_list(cand["legal_tilings"], workers_for_cand)
        for partition in partitions:
            if partition:
                work_items.append((
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
                ))

    print(f"  {operator.name} (B={operator.B},M={operator.M},K={operator.K},N={operator.N}): "
          f"{total_tilings} chip simulations across {len(candidates)} candidates, "
          f"dispatched to {len(work_items)} workers")

    # Phase 3: Execute all work items
    # Use 'spawn' context to avoid nested-fork deadlocks when this function is
    # called from inside a forked child process (e.g. multiprocessing.Process).
    # 'fork' inherits dead threads and corrupted C++ mutex state; 'spawn' starts
    # fresh interpreters that are safe to nest.
    if num_workers <= 1:
        all_results = [explore_gemm_comp_worker(*item) for item in work_items]
    else:
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(num_workers, initializer=set_pdeathsig) as pool:
            all_results = pool.starmap(explore_gemm_comp_worker, work_items)

    # Phase 4: Find best tiling per candidate, then global best
    candidate_best = {}
    for result in all_results:
        cand_id = result["candidate_id"]
        if cand_id not in candidate_best or result["latency"] < candidate_best[cand_id]["latency"]:
            candidate_best[cand_id] = result

    opt_latency = float('inf')
    opt_comp_description = None
    for cand_id, best_result in candidate_best.items():
        cand = candidates[cand_id]
        channel_B = cand["channel_B"]
        total_channel_latency = best_result["latency"] * channel_B
        inter_channel_comm_latency = cand["inter_channel_communication_latency"]
        if total_channel_latency + inter_channel_comm_latency < opt_latency:
            opt_latency = total_channel_latency + inter_channel_comm_latency
            core_placement_result = _generate_gemm_data_placements(cand["core_operator"], element_size, dram_row_size, base_addr)
            core_input = core_placement_result["input_placement"]
            core_output = core_placement_result["output_placement"]
            core_weights = core_placement_result["weight_placement_list"]
            core_next_addr = core_placement_result["next_base_addr"]
            opt_comp_description = {
                "name": operator.name,
                "description": best_result["description"],
                "gemm_num_per_channel": best_result["gemm_num"],
                "core_tiling_factors": best_result["tiling_factors"],
                "intra_channel_tiling_factors": (cand["core_nK"], cand["core_nN"]),
                "inter_channel_tiling_factors": (cand["channel_nK"], cand["channel_nN"]),
                "channel_group_num": cand["channel_num_factor"],
                "computation_latency": best_result["computation_latency"] * channel_B,
                "intra_channel_accumulation_latency": best_result["intra_channel_accumulation_latency"] * channel_B,
                "intra_channel_accumulation_energy": best_result["intra_channel_accumulation_energy"] * channel_B,
                "inter_channel_communication_latency": inter_channel_comm_latency,
                "inter_channel_communication_energy": cand["inter_channel_communication_energy"],
                "latency": total_channel_latency + inter_channel_comm_latency,
                "next_base_addr": core_next_addr,
                "input_placement": core_input,
                "output_placement": core_output,
                "weight_placement_list": core_weights,
                "core_M": cand["core_operator"].M,
                "core_K": cand["core_operator"].K,
                "core_N": cand["core_operator"].N,
                "channel_B": channel_B,
            }

    if opt_comp_description is not None:
        cache_dir = os.path.join(cache_base_dir, operator.name)
        os.makedirs(cache_dir, exist_ok=True)
        cache_data = {
            "M": operator.M,
            "K": operator.K,
            "N": operator.N,
            "B": operator.B,
            "element_size": element_size,
            "core_M": int(opt_comp_description["core_M"]),
            "core_K": int(opt_comp_description["core_K"]),
            "core_N": int(opt_comp_description["core_N"]),
            "channel_B": int(opt_comp_description["channel_B"]),
            "description": opt_comp_description["description"],
            "gemm_num_per_channel": opt_comp_description["gemm_num_per_channel"],
            "core_tiling_factors": list(opt_comp_description["core_tiling_factors"]),
            "intra_channel_tiling_factors": list(opt_comp_description["intra_channel_tiling_factors"]),
            "inter_channel_tiling_factors": list(opt_comp_description["inter_channel_tiling_factors"]),
            "channel_group_num": opt_comp_description["channel_group_num"],
            "computation_latency": float(opt_comp_description["computation_latency"]),
            "intra_channel_accumulation_latency": float(opt_comp_description["intra_channel_accumulation_latency"]),
            "intra_channel_accumulation_energy": float(opt_comp_description["intra_channel_accumulation_energy"]),
            "inter_channel_communication_latency": float(opt_comp_description["inter_channel_communication_latency"]),
            "inter_channel_communication_energy": float(opt_comp_description["inter_channel_communication_energy"]),
            "latency": float(opt_comp_description["latency"]),
        }
        with open(cache_file_path, "w") as f:
            yaml.dump(cache_data, f)

    return opt_comp_description


def explore_edge_tiling(
    # Operator-related configs
    model_config: ModelConfig,
    attention_block: List[Operator],
    ffn_moe_block: List[Operator],
    # Input-related configs
    context_length_list: List[int],
    # Hardware-related configs
    edge_config: EdgeSystemConfig,
    # Some hyper parameters
    element_size: int = 2,
    dram_row_size: int = 16*1024,
    min_tM: int = 8,
    min_tK: int = 128,
    min_tN: int = 8,
    num_workers: int = int(0.8 * multiprocessing.cpu_count()),
    # Intermediate results storing configs
    intermediate_result_dir: str = "",
    # Shared GEMM tiling cache directory (reusable across context lengths)
    gemm_tiling_cache_dir: str = "",
):
    if intermediate_result_dir == "":
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        channel_num = edge_config.channel_num
        core_num = edge_config.chip_config.core_num
        intermediate_result_dir = os.path.join(
            project_root,
            f"decaparated/edge_tiling_exploration/{model_config.name}_{channel_num}channels_{core_num}cores"
        )
    os.makedirs(intermediate_result_dir, exist_ok=True)

    set_shape(
        model_config=model_config,
        attention_block=attention_block,
        ffn_moe_block=ffn_moe_block,
        context_length_list=context_length_list,
    )
    full_operator_list = attention_block + ffn_moe_block

    # Each core's data placement is related with tiling exploration in edge accelerators,
    # so we need to generate data placement on the fly during tiling exploration.
    base_addr = 0
    data_placement_list = []

    task_description_list = []
    inter_chip_communication_list = []

    for i, operator in enumerate[Operator](full_operator_list):
        print(f"Processing operator {operator.name}")
        
        if operator.name == "softmax":
            inter_chip_communication_list.append(operator)
            continue

        if operator.op_type == OperatorType.GEMM:
            comp_desctiption = explore_gemm_comp(
                # Tensor description
                base_addr=base_addr,
                # Operator shape description
                operator=operator,
                element_size=element_size,
                # Edge system config
                edge_config=edge_config,
                # Hyper parameters
                dram_row_size=dram_row_size,
                min_tM=min_tM,
                min_tK=min_tK,
                min_tN=min_tN,
                num_workers=num_workers,
                # Intermediate result directory
                intermediate_result_dir=intermediate_result_dir,
                gemm_tiling_cache_dir=gemm_tiling_cache_dir,
            )

            data_placement_list.append(comp_desctiption["input_placement"])
            data_placement_list.append(comp_desctiption["output_placement"])
            data_placement_list.extend(comp_desctiption["weight_placement_list"])
            task_description_list.append(("gemm", comp_desctiption))

            base_addr = comp_desctiption["next_base_addr"]
        else:
            assert False, f"Unsupported operator: name {operator.name}, type {operator.op_type}"

    return {
        "data_placement_list": data_placement_list,
        "task_description_list": task_description_list,
        "inter_chip_communication_list": inter_chip_communication_list,
    }
