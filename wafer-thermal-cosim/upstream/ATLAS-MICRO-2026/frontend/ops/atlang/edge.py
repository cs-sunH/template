import multiprocessing
from typing import Dict, Any, List

import frontend.atlang.language as A

from frontend.hardware_parser import EdgeSystemConfig
from frontend.model_parser import Operator


# Atlang is a simulator-only DSL frontend. The decorated kernel returns
# an AtlangKernel shell populated by AST capture and simulator extraction.
def edge_inference(
    # Basic configs
    edge_config: EdgeSystemConfig,
    # Operator shape description
    operator_dict: Dict[str, Dict[str, Any]],
    inter_chip_communication_list: List[Operator],
    # Data type
    dtype=A.float16,
    # Helper configs
    intermediate_result_dir: str = "",
    gemm_tiling_cache_dir: str = "",
    num_layers: int = 1,
    num_workers: int = int(0.8 * multiprocessing.cpu_count()),
):
    core_num = edge_config.chip_config.core_num

    min_tM = 8
    min_tK = 128
    min_tN = 8
    # This edge kernel defines attention_qk as (kv_group, head_dim, context) with gemm_b = batch * kv_head.
    qkv_proj_shape = tuple(int(dim) for dim in operator_dict["qkv_proj"]["gemm_shape"])
    attention_qk_shape = tuple(int(dim) for dim in operator_dict["attention_qk"]["gemm_shape"])
    edge_softmax_batch_size = qkv_proj_shape[0]
    edge_softmax_kv_group_num = attention_qk_shape[0]
    edge_softmax_context_length = attention_qk_shape[2]
    edge_softmax_attention_batch = int(operator_dict["attention_qk"]["gemm_b"])
    if edge_softmax_batch_size <= 0 or edge_softmax_attention_batch % edge_softmax_batch_size != 0:
        raise ValueError(
            "Edge softmax metadata requires attention_qk gemm_b to be divisible by qkv_proj batch size, "
            f"but got attention_qk gemm_b={edge_softmax_attention_batch} and batch size={edge_softmax_batch_size}."
        )
    edge_softmax_kv_head_num = edge_softmax_attention_batch // edge_softmax_batch_size
    core_array_kwargs = {
        # Hardware configs
        "system_config": edge_config,
        "dram_row_size": 16*1024,
        # Operator configs
        "min_tM": min_tM,
        "min_tK": min_tK,
        "min_tN": min_tN,
        "inter_chip_communication_list": inter_chip_communication_list,
        "num_layers": num_layers,
        "edge_softmax_attention_operator_name": "attention_qk",
        "edge_softmax_batch_size": edge_softmax_batch_size,
        "edge_softmax_context_length": edge_softmax_context_length,
        "edge_softmax_kv_group_num": edge_softmax_kv_group_num,
        "edge_softmax_kv_head_num": edge_softmax_kv_head_num,
        # Explorer configs
        "num_workers": num_workers,
        "intermediate_result_dir": intermediate_result_dir,
        "gemm_tiling_cache_dir": gemm_tiling_cache_dir,
    }

    # Computation kwargs
    # (1) QKV projection
    qkv_proj_partition_kwargs = {
        "gemm_shape": operator_dict["qkv_proj"]["gemm_shape"],
        "gemm_b": operator_dict["qkv_proj"]["gemm_b"],
        # Special for edge inference, since H2LLM automatically explores inter-chip & inter-core mapping,
        # when all dim splitting mappings are None, we follow H2LLM's setup to automatically explore partition.
        "core_dim_mapping": (None, None, None)
    }
    # (2) Attention QK
    attention_qk_partition_kwargs = {
        "gemm_shape": operator_dict["attention_qk"]["gemm_shape"],
        "gemm_b": operator_dict["attention_qk"]["gemm_b"],
        "core_dim_mapping": (None, None, None)
    }
    # (3) Attention SV
    attention_sv_partition_kwargs = {
        "gemm_shape": operator_dict["attention_sv"]["gemm_shape"],
        "gemm_b": operator_dict["attention_sv"]["gemm_b"],
        "core_dim_mapping": (None, None, None)
    }
    # (4) O projection
    o_proj_partition_kwargs = {
        "gemm_shape": operator_dict["o_proj"]["gemm_shape"],
        "gemm_b": operator_dict["o_proj"]["gemm_b"],
        "core_dim_mapping": (None, None, None)
    }
    # (5) Down projection
    down_proj_partition_kwargs = {
        "gemm_shape": operator_dict["down_proj"]["gemm_shape"],
        "gemm_b": operator_dict["down_proj"]["gemm_b"],
        "core_dim_mapping": (None, None, None)
    }
    # (6) Up projection
    up_proj_partition_kwargs = {
        "gemm_shape": operator_dict["up_proj"]["gemm_shape"],
        "gemm_b": operator_dict["up_proj"]["gemm_b"],
        "core_dim_mapping": (None, None, None)
    }

    # When conducting edge kernel mapping exploration, we automatically estimate inter-chip communication overhead
    # for data preparation/sccumulation before/after each operator. Therefore, we do not need to define inter-chip
    # communication kernel before/after each GEMM kernel.
    #
    # Besides, since edge chip does not have inter-core NoC in each chip, so we do not need to design communication
    # kernels here.
    @A.main
    def inference(
        input_qkv_proj: A.Tensor,
        weight_qkv_proj: A.Tensor,
        output_qkv_proj: A.Tensor,
        input_attention_qk: A.Tensor,
        weight_attention_qk: A.Tensor,
        output_attention_qk: A.Tensor,
        input_attention_sv: A.Tensor,
        weight_attention_sv: A.Tensor,
        output_attention_sv: A.Tensor,
        input_o_proj: A.Tensor,
        weight_o_proj: A.Tensor,
        output_o_proj: A.Tensor,
        input_down_proj: A.Tensor,
        weight_down_proj: A.Tensor,
        output_down_proj: A.Tensor,
        input_up_proj: A.Tensor,
        weight_up_proj: A.Tensor,
        output_up_proj: A.Tensor,
    ):
        with A.CoreArray(shape=(core_num,), **core_array_kwargs):
            with A.SPMD(name="qkv_proj", type="gemm", **qkv_proj_partition_kwargs) as (core_M, core_K, core_N):
                with A.Kernel(A.ceildiv(core_M, min_tM), A.ceildiv(core_K, min_tK), A.ceildiv(core_N, min_tN), autotune=True) as (bM, bK, bN):
                    input_qkv_proj = A.Tensor(shape=(core_M, core_K), strides=(core_K, 1), dtype=dtype)
                    weight_qkv_proj = A.Tensor(shape=(core_K, core_N), strides=(1, core_K), dtype=dtype)
                    output_qkv_proj = A.Tensor(shape=(core_M, core_N), strides=(core_N, 1), dtype=dtype)

                    layer_input_tile = A.alloc(shape=(min_tM, min_tK), dtype=dtype)
                    qkv_proj_weight_tile = A.alloc(shape=(min_tK, min_tN), dtype=dtype)
                    qkv_proj_output_tile = A.alloc(shape=(min_tM, min_tN), dtype=dtype)
                    qkv_proj_output_acc = A.alloc(shape=(min_tM, min_tN), dtype=dtype)

                    for m in A.Serial(bM):
                        for n in A.Serial(bN):
                            A.clear(qkv_proj_output_tile)
                            for k in A.Serial(bK):
                                A.copy(input_qkv_proj[m * min_tM, k * min_tK], layer_input_tile)
                                A.copy(weight_qkv_proj[k * min_tK, n * min_tN], qkv_proj_weight_tile)
                                A.gemm(layer_input_tile, qkv_proj_weight_tile, qkv_proj_output_acc)
                                A.add(qkv_proj_output_acc, qkv_proj_output_tile, qkv_proj_output_tile) # Newly added element-wise vector ops
                            A.copy(qkv_proj_output_tile, output_qkv_proj[m * min_tM, n * min_tN])
            
            with A.SPMD(name="attention_qk", type="gemm", **attention_qk_partition_kwargs) as (core_M, core_K, core_N):
                with A.Kernel(A.ceildiv(core_M, min_tM), A.ceildiv(core_K, min_tK), A.ceildiv(core_N, min_tN), autotune=True) as (bM, bK, bN):
                    input_attention_qk = A.Tensor(shape=(core_M, core_K), strides=(core_K, 1), dtype=dtype)
                    weight_attention_qk = A.Tensor(shape=(core_K, core_N), strides=(1, core_K), dtype=dtype)
                    output_attention_qk = A.Tensor(shape=(core_M, core_N), strides=(core_N, 1), dtype=dtype)

                    attention_qk_input_tile = A.alloc(shape=(min_tM, min_tK), dtype=dtype)
                    attention_qk_weight_tile = A.alloc(shape=(min_tK, min_tN), dtype=dtype)
                    attention_qk_output_tile = A.alloc(shape=(min_tM, min_tN), dtype=dtype)
                    attention_qk_output_acc = A.alloc(shape=(min_tM, min_tN), dtype=dtype)

                    for m in A.Serial(bM):
                        for n in A.Serial(bN):
                            A.clear(attention_qk_output_tile)
                            for k in A.Serial(bK):
                                A.copy(input_attention_qk[m * min_tM, k * min_tK], attention_qk_input_tile)
                                A.copy(weight_attention_qk[k * min_tK, n * min_tN], attention_qk_weight_tile)
                                A.gemm(attention_qk_input_tile, attention_qk_weight_tile, attention_qk_output_acc)
                                A.add(attention_qk_output_acc, attention_qk_output_tile, attention_qk_output_tile) # Newly added element-wise vector ops
                            A.copy(attention_qk_output_tile, output_attention_qk[m * min_tM, n * min_tN])

            with A.SPMD(name="attention_sv", type="gemm", **attention_sv_partition_kwargs) as (core_M, core_K, core_N):
                with A.Kernel(A.ceildiv(core_M, min_tM), A.ceildiv(core_K, min_tK), A.ceildiv(core_N, min_tN), autotune=True) as (bM, bK, bN):
                    input_attention_sv = A.Tensor(shape=(core_M, core_K), strides=(core_K, 1), dtype=dtype)
                    weight_attention_sv = A.Tensor(shape=(core_K, core_N), strides=(1, core_K), dtype=dtype)
                    output_attention_sv = A.Tensor(shape=(core_M, core_N), strides=(core_N, 1), dtype=dtype)

                    attention_sv_input_tile = A.alloc(shape=(min_tM, min_tK), dtype=dtype)
                    attention_sv_weight_tile = A.alloc(shape=(min_tK, min_tN), dtype=dtype)
                    attention_sv_output_tile = A.alloc(shape=(min_tM, min_tN), dtype=dtype)
                    attention_sv_output_acc = A.alloc(shape=(min_tM, min_tN), dtype=dtype)

                    for m in A.Serial(bM):
                        for n in A.Serial(bN):
                            A.clear(attention_sv_output_tile)
                            for k in A.Serial(bK):
                                A.copy(input_attention_sv[m * min_tM, k * min_tK], attention_sv_input_tile)
                                A.copy(weight_attention_sv[k * min_tK, n * min_tN], attention_sv_weight_tile)
                                A.gemm(attention_sv_input_tile, attention_sv_weight_tile, attention_sv_output_acc)
                                A.add(attention_sv_output_acc, attention_sv_output_tile, attention_sv_output_tile) # Newly added element-wise vector ops
                            A.copy(attention_sv_output_tile, output_attention_sv[m * min_tM, n * min_tN])

            with A.SPMD(name="o_proj", type="gemm", **o_proj_partition_kwargs) as (core_M, core_K, core_N):
                with A.Kernel(A.ceildiv(core_M, min_tM), A.ceildiv(core_K, min_tK), A.ceildiv(core_N, min_tN), autotune=True) as (bM, bK, bN):
                    input_o_proj = A.Tensor(shape=(core_M, core_K), strides=(core_K, 1), dtype=dtype)
                    weight_o_proj = A.Tensor(shape=(core_K, core_N), strides=(1, core_K), dtype=dtype)
                    output_o_proj = A.Tensor(shape=(core_M, core_N), strides=(core_N, 1), dtype=dtype)

                    o_proj_input_tile = A.alloc(shape=(min_tM, min_tK), dtype=dtype)
                    o_proj_weight_tile = A.alloc(shape=(min_tK, min_tN), dtype=dtype)
                    o_proj_output_tile = A.alloc(shape=(min_tM, min_tN), dtype=dtype)
                    o_proj_output_acc = A.alloc(shape=(min_tM, min_tN), dtype=dtype)

                    for m in A.Serial(bM):
                        for n in A.Serial(bN):
                            A.clear(o_proj_output_tile)
                            for k in A.Serial(bK):
                                A.copy(input_o_proj[m * min_tM, k * min_tK], o_proj_input_tile)
                                A.copy(weight_o_proj[k * min_tK, n * min_tN], o_proj_weight_tile)
                                A.gemm(o_proj_input_tile, o_proj_weight_tile, o_proj_output_acc)
                                A.add(o_proj_output_acc, o_proj_output_tile, o_proj_output_tile) # Newly added element-wise vector ops
                            A.copy(o_proj_output_tile, output_o_proj[m * min_tM, n * min_tN])

            with A.SPMD(name="down_proj", type="gemm", **down_proj_partition_kwargs) as (core_M, core_K, core_N):
                with A.Kernel(A.ceildiv(core_M, min_tM), A.ceildiv(core_K, min_tK), A.ceildiv(core_N, min_tN), autotune=True) as (bM, bK, bN):
                    input_down_proj = A.Tensor(shape=(core_M, core_K), strides=(core_K, 1), dtype=dtype)
                    weight_down_proj = A.Tensor(shape=(core_K, core_N), strides=(1, core_K), dtype=dtype)
                    output_down_proj = A.Tensor(shape=(core_M, core_N), strides=(core_N, 1), dtype=dtype)

                    down_proj_input_tile = A.alloc(shape=(min_tM, min_tK), dtype=dtype)
                    down_proj_weight_tile = A.alloc(shape=(min_tK, min_tN), dtype=dtype)
                    down_proj_output_tile = A.alloc(shape=(min_tM, min_tN), dtype=dtype)
                    down_proj_output_acc = A.alloc(shape=(min_tM, min_tN), dtype=dtype)

                    for m in A.Serial(bM):
                        for n in A.Serial(bN):
                            A.clear(down_proj_output_tile)
                            for k in A.Serial(bK):
                                A.copy(input_down_proj[m * min_tM, k * min_tK], down_proj_input_tile)
                                A.copy(weight_down_proj[k * min_tK, n * min_tN], down_proj_weight_tile)
                                A.gemm(down_proj_input_tile, down_proj_weight_tile, down_proj_output_acc)
                                A.add(down_proj_output_acc, down_proj_output_tile, down_proj_output_tile) # Newly added element-wise vector ops
                            A.copy(down_proj_output_tile, output_down_proj[m * min_tM, n * min_tN])

            with A.SPMD(name="up_proj", type="gemm", **up_proj_partition_kwargs) as (core_M, core_K, core_N):
                with A.Kernel(A.ceildiv(core_M, min_tM), A.ceildiv(core_K, min_tK), A.ceildiv(core_N, min_tN), autotune=True) as (bM, bK, bN):
                    input_up_proj = A.Tensor(shape=(core_M, core_K), strides=(core_K, 1), dtype=dtype)
                    weight_up_proj = A.Tensor(shape=(core_K, core_N), strides=(1, core_K), dtype=dtype)
                    output_up_proj = A.Tensor(shape=(core_M, core_N), strides=(core_N, 1), dtype=dtype)

                    up_proj_input_tile = A.alloc(shape=(min_tM, min_tK), dtype=dtype)
                    up_proj_weight_tile = A.alloc(shape=(min_tK, min_tN), dtype=dtype)
                    up_proj_output_tile = A.alloc(shape=(min_tM, min_tN), dtype=dtype)
                    up_proj_output_acc = A.alloc(shape=(min_tM, min_tN), dtype=dtype)

                    for m in A.Serial(bM):
                        for n in A.Serial(bN):
                            A.clear(up_proj_output_tile)
                            for k in A.Serial(bK):
                                A.copy(input_up_proj[m * min_tM, k * min_tK], up_proj_input_tile)
                                A.copy(weight_up_proj[k * min_tK, n * min_tN], up_proj_weight_tile)
                                A.gemm(up_proj_input_tile, up_proj_weight_tile, up_proj_output_acc)
                                A.add(up_proj_output_acc, up_proj_output_tile, up_proj_output_tile) # Newly added element-wise vector ops
                            A.copy(up_proj_output_tile, output_up_proj[m * min_tM, n * min_tN])

    return inference
