import math
import multiprocessing
from typing import List, Dict, Any

import frontend.atlang.language as A

from frontend.hardware_parser import CloudSystemConfig, extract_noc_topology
from frontend.model_parser import Operator
from frontend.util import (
    prepare_kv_comm_workload_size, 
    generate_mesh_ring_topology,
)


def _line_neighbor(core_array_size: int, axis: int, core_id, delta: int):
    if axis == 0:
        return core_id + delta * core_array_size
    if axis == 1:
        return core_id + delta
    raise ValueError(f"Unsupported line axis {axis}")


def _copy_line_chunk(
    tensor,
    buffer,
    *,
    row_num: int,
    chunk_length: int,
    chunk_id,
    base_offset=0,
):
    A.copy(
        tensor[
            : row_num,
            base_offset + chunk_id * chunk_length: base_offset + (chunk_id + 1) * chunk_length,
        ],
        buffer,
    )


def _store_line_chunk(
    buffer,
    tensor,
    *,
    row_num: int,
    chunk_length: int,
    chunk_id,
    base_offset=0,
):
    A.copy(
        buffer,
        tensor[
            : row_num,
            base_offset + chunk_id * chunk_length: base_offset + (chunk_id + 1) * chunk_length,
        ],
    )


def _store_kv_cache_vector(
    buffer,
    kv_cache,
    *,
    src_row,
    head_idx,
    slot_id: int,
    head_dim: int,
    total_slot_num: int,
):
    A.copy(
        buffer[
            src_row: (src_row + 1), 
            head_idx * head_dim: (head_idx + 1) * head_dim,
        ],
        kv_cache[
            head_idx * total_slot_num + slot_id: head_idx * total_slot_num + slot_id + 1,
            :,
        ],
    )


def sanity_check(cloud_config: CloudSystemConfig):
    # Sanity check. This operator implementation is customized for 4*4 2D-mesh
    # NOTE: KV comm's MPMD all reduce communication pattern is highly related to mesh size,
    #       so we constrain the mesh size to be 4*4 here.
    topology, k, n = extract_noc_topology(cloud_config.chip_config.noc_config.config_path)
    if topology != "mesh":
        raise ValueError("This operator implementation is customized for 4*4 2D-mesh.")
    core_num = cloud_config.chip_config.core_num
    core_array_size = int(math.sqrt(core_num))
    if core_num != 16 and core_array_size != 4:
        raise ValueError("This operator implementation is customized for 4*4 2D-mesh.")


# Atlang is a simulator-only DSL frontend. The decorated kernel returns
# an AtlangKernel shell populated by AST capture and simulator extraction.
def cloud_inference(
    # Basic configs
    cloud_config: CloudSystemConfig,
    # Operator shape description
    operator_dict: Dict[str, Dict[str, Any]],
    inter_chip_communication_list: List[Operator],
    # KV cache placement description
    context_slot_mapping: List[Dict[int, List[int]]],
    last_slot_mapping: List[List[List[int]]],
    # Data type
    dtype=A.float16,
    # Helper configs
    intermediate_result_dir: str = "",
    gemm_tiling_cache_dir: str = "",
    num_layers: int = 1,
    num_workers: int = int(0.8 * multiprocessing.cpu_count()),
):
    core_num = cloud_config.chip_config.core_num
    core_array_size = int(math.sqrt(core_num))
    
    # Global kwargs
    min_tM = 8
    min_tK = 512
    min_tN = 8
    min_tS = 64
    core_array_kwargs = {
        # Hardware configs
        "system_config": cloud_config,
        "dram_row_size": 128*1024,
        "flit_size": cloud_config.chip_config.noc_config.flit_size,
        # Operator configs
        "min_tM": min_tM,
        "min_tK": min_tK,
        "min_tN": min_tN,
        "min_tS": min_tS,
        "inter_chip_communication_list": inter_chip_communication_list,
        "num_layers": num_layers,
        # Explorer configs
        "num_workers": num_workers,
        "intermediate_result_dir": intermediate_result_dir,
        "gemm_tiling_cache_dir": gemm_tiling_cache_dir,
    }

    # Computation kwargs
    # (1) QKV projection
    # (M, K) × (K, N) -> (M, N)
    # both gemm_shape and core_dim_mapping are listed in the order of (M, K, N)
    qkv_proj_partition_kwargs = {
        "gemm_shape": operator_dict["qkv_proj"]["gemm_shape"],
        "gemm_b": operator_dict["qkv_proj"]["gemm_b"],
        "core_dim_mapping": (None, (0,), (1,))
    }
    # (2) Attention
    attention_kwargs = {
        "attention_shape": operator_dict["attention"]["attention_shape"],
        "context_slot_mapping": context_slot_mapping,
    }
    # (3) O projection
    o_proj_partition_kwargs = {
        "gemm_shape": operator_dict["o_proj"]["gemm_shape"],
        "gemm_b": operator_dict["o_proj"]["gemm_b"],
        "core_dim_mapping": (None, (0,), (1,))
    }
    # (4) Down projection
    down_proj_partition_kwargs = {
        "gemm_shape": operator_dict["down_proj"]["gemm_shape"],
        "gemm_b": operator_dict["down_proj"]["gemm_b"],
        "core_dim_mapping": (None, (1,), (0,))
    }
    # (5) Up projection
    up_proj_partition_kwargs = {
        "gemm_shape": operator_dict["up_proj"]["gemm_shape"],
        "gemm_b": operator_dict["up_proj"]["gemm_b"],
        "core_dim_mapping": (None, (0,), (1,))
    }

    # Communication_kwargs
    # (1) q/kv_proj_comm
    qkv_comm_row_num = operator_dict["qkv_proj"]["gemm_shape"][0]
    kv_head_num = operator_dict["attention"]["attention_shape"][1]
    q_head_num = kv_head_num * operator_dict["attention"]["attention_shape"][2]
    q_length = int(math.ceil(operator_dict["qkv_proj"]["gemm_shape"][2] / core_array_size) * q_head_num / (q_head_num + kv_head_num*2))
    q_all_reduce_chunk_length = math.ceil(q_length / core_array_size)
    q_all_gather_chunk_length = q_length
    # (2) kv_proj_comm
    kv_length = math.ceil(operator_dict["qkv_proj"]["gemm_shape"][2] / core_array_size) - q_length
    total_slot_num = math.ceil(cloud_config.max_context_length / core_num)
    num_kv_heads = operator_dict["attention"]["attention_shape"][1]
    head_dim = operator_dict["attention"]["attention_shape"][4]
    (
        x_reduce_first_half_req_num,
        x_reduce_second_half_req_num,
        x_scatter_first_half_per_hop_req_num,
        x_scatter_second_half_per_hop_req_num,
        y_gather_first_half_req_num,
        y_gather_second_half_req_num,
        y_scatter_first_half_per_hop_req_num,
        y_scatter_second_half_per_hop_req_num,
    ) = prepare_kv_comm_workload_size(
        core_array_size=core_array_size,
        last_slot_mapping=last_slot_mapping,
    )
    # (3) attention_comm
    rank_core_map = generate_mesh_ring_topology(core_array_size, return_rank_to_core=True)
    core_rank_map = generate_mesh_ring_topology(core_array_size, return_rank_to_core=False)
    attn_output_hidden_dim = operator_dict["attention"]["attention_shape"][1] \
                           * operator_dict["attention"]["attention_shape"][2] \
                           * operator_dict["attention"]["attention_shape"][4]
    attention_comm_row_num = operator_dict["attention"]["attention_shape"][0]
    attn_reduce_scatter_chunk_length = math.ceil(attn_output_hidden_dim / core_num)
    attn_all_gather_chunk_length = math.ceil(math.ceil(operator_dict["o_proj"]["gemm_shape"][1] / core_array_size) / core_array_size)
    # (4) o_proj_comm
    o_proj_comm_row_num = operator_dict["o_proj"]["gemm_shape"][0]
    o_all_reduce_chunk_length = math.ceil(math.ceil(operator_dict["o_proj"]["gemm_shape"][2] / core_array_size) / core_array_size)
    # (5) down_proj_comm
    down_proj_comm_row_num = operator_dict["down_proj"]["gemm_shape"][0]
    down_all_reduce_chunk_length = math.ceil(math.ceil(operator_dict["down_proj"]["gemm_shape"][2] / core_array_size) / core_array_size)
    # (6) up_proj_comm
    up_proj_comm_row_num = operator_dict["up_proj"]["gemm_shape"][0]
    up_all_reduce_chunk_length = math.ceil(math.ceil(operator_dict["qkv_proj"]["gemm_shape"][1] / core_array_size) / core_array_size)

    # When conducting cloud kernel mapping exploration, we automatically estimate inter-chip TP/EP communication 
    # overhead. Therefore, we do not need to define inter-chip collective communication kernels.
    @A.main
    def inference(
        input_qkv_proj: A.Tensor,
        weight_qkv_proj: A.Tensor,
        output_qkv_proj: A.Tensor,
        input_attention: A.Tensor,
        kv_cache: A.Tensor,
        output_attention: A.Tensor,
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
        with A.CoreArray(shape=(core_array_size, core_array_size), **core_array_kwargs):
            # gemm_shape is sorted in (M, K, N) order for (M, K) × (K, N) -> (M, N)
            # core_dim_mapping is also sorted in (M, K, N) order, each item represents which CoreArray dimensions are used to split corresponding GEMM
            # For example, core_dim_mapping = (None, (0,), (1,)) means that M is not split, K is split by CoreArray.shape[0], N is split by CoreArray.shape[1]
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
            
            # CORE_ID is a special variable, representing each kernel's core ID under MPMD programming model
            row_num = qkv_comm_row_num
            with A.MPMD(name="qkv_proj_comm_q_all_reduce", type="communication") as CORE_ID:
                with A.Kernel(core_list=list(range(core_num))):
                    q_all_reduce_line_pos = CORE_ID // core_array_size
                    pos_wave_buffer = A.alloc(shape=(row_num, q_all_reduce_chunk_length), dtype=dtype)
                    neg_wave_buffer = A.alloc(shape=(row_num, q_all_reduce_chunk_length), dtype=dtype)
                    recv_from_neg = A.alloc(shape=(row_num, q_all_reduce_chunk_length), dtype=dtype)
                    recv_from_pos = A.alloc(shape=(row_num, q_all_reduce_chunk_length), dtype=dtype)
                    owned_chunk_buffer = A.alloc(shape=(row_num, q_all_reduce_chunk_length), dtype=dtype)

                    for t in A.Serial(core_array_size - 1):
                        if q_all_reduce_line_pos == 0:
                            _copy_line_chunk(
                                output_qkv_proj,
                                pos_wave_buffer,
                                row_num=row_num,
                                chunk_length=q_all_reduce_chunk_length,
                                chunk_id=t,
                            )
                            A.send(CORE_ID, _line_neighbor(core_array_size, 0, CORE_ID, +1), pos_wave_buffer)
                        if 0 < q_all_reduce_line_pos < core_array_size - 1 and q_all_reduce_line_pos <= t:
                            A.send(CORE_ID, _line_neighbor(core_array_size, 0, CORE_ID, +1), pos_wave_buffer)
                        if q_all_reduce_line_pos > 0 and q_all_reduce_line_pos - 1 <= t:
                            _copy_line_chunk(
                                output_qkv_proj,
                                pos_wave_buffer,
                                row_num=row_num,
                                chunk_length=q_all_reduce_chunk_length,
                                chunk_id=t - (q_all_reduce_line_pos - 1),
                            )
                            A.recv(_line_neighbor(core_array_size, 0, CORE_ID, -1), CORE_ID, recv_from_neg)
                            A.add(pos_wave_buffer, recv_from_neg, pos_wave_buffer)

                        if q_all_reduce_line_pos == core_array_size - 1:
                            _copy_line_chunk(
                                output_qkv_proj,
                                neg_wave_buffer,
                                row_num=row_num,
                                chunk_length=q_all_reduce_chunk_length,
                                chunk_id=core_array_size - 1 - t,
                            )
                            A.send(CORE_ID, _line_neighbor(core_array_size, 0, CORE_ID, -1), neg_wave_buffer)
                        if 0 < q_all_reduce_line_pos < core_array_size - 1 and q_all_reduce_line_pos >= core_array_size - 1 - t:
                            A.send(CORE_ID, _line_neighbor(core_array_size, 0, CORE_ID, -1), neg_wave_buffer)
                        if q_all_reduce_line_pos < core_array_size - 1 and q_all_reduce_line_pos + 1 >= core_array_size - 1 - t:
                            _copy_line_chunk(
                                output_qkv_proj,
                                neg_wave_buffer,
                                row_num=row_num,
                                chunk_length=q_all_reduce_chunk_length,
                                chunk_id=2 * core_array_size - 3 - q_all_reduce_line_pos - t,
                            )
                            A.recv(_line_neighbor(core_array_size, 0, CORE_ID, +1), CORE_ID, recv_from_pos)
                            A.add(neg_wave_buffer, recv_from_pos, neg_wave_buffer)
                        if t == core_array_size - 2:
                            if q_all_reduce_line_pos == 0:
                                A.copy(neg_wave_buffer, owned_chunk_buffer)
                            elif q_all_reduce_line_pos == core_array_size - 1:
                                A.copy(pos_wave_buffer, owned_chunk_buffer)
                            elif q_all_reduce_line_pos < core_array_size // 2:
                                A.add(pos_wave_buffer, recv_from_pos, owned_chunk_buffer)
                            else:
                                A.add(recv_from_neg, neg_wave_buffer, owned_chunk_buffer)

                    for t in A.Serial(core_array_size - 1):
                        if t == 0:
                            _store_line_chunk(
                                owned_chunk_buffer,
                                input_attention,
                                row_num=row_num,
                                chunk_length=q_all_reduce_chunk_length,
                                chunk_id=core_array_size - 1 - q_all_reduce_line_pos,
                                base_offset=(CORE_ID % core_array_size) * q_length,
                            )
                            if q_all_reduce_line_pos > 0:
                                A.send(CORE_ID, _line_neighbor(core_array_size, 0, CORE_ID, -1), owned_chunk_buffer)
                            if q_all_reduce_line_pos < core_array_size - 1:
                                A.send(CORE_ID, _line_neighbor(core_array_size, 0, CORE_ID, +1), owned_chunk_buffer)

                        if q_all_reduce_line_pos < core_array_size - 1 and t > 0 and t <= q_all_reduce_line_pos:
                            A.send(CORE_ID, _line_neighbor(core_array_size, 0, CORE_ID, +1), recv_from_neg)
                        if q_all_reduce_line_pos > 0 and t > 0 and t <= core_array_size - 1 - q_all_reduce_line_pos:
                            A.send(CORE_ID, _line_neighbor(core_array_size, 0, CORE_ID, -1), recv_from_pos)

                        if q_all_reduce_line_pos > 0 and t < q_all_reduce_line_pos:
                            A.recv(_line_neighbor(core_array_size, 0, CORE_ID, -1), CORE_ID, recv_from_neg)
                            _store_line_chunk(
                                recv_from_neg,
                                input_attention,
                                row_num=row_num,
                                chunk_length=q_all_reduce_chunk_length,
                                chunk_id=core_array_size - q_all_reduce_line_pos + t,
                                base_offset=(CORE_ID % core_array_size) * q_length,
                            )
                        if q_all_reduce_line_pos < core_array_size - 1 and t < core_array_size - 1 - q_all_reduce_line_pos:
                            A.recv(_line_neighbor(core_array_size, 0, CORE_ID, +1), CORE_ID, recv_from_pos)
                            _store_line_chunk(
                                recv_from_pos,
                                input_attention,
                                row_num=row_num,
                                chunk_length=q_all_reduce_chunk_length,
                                chunk_id=core_array_size - q_all_reduce_line_pos - 2 - t,
                                base_offset=(CORE_ID % core_array_size) * q_length,
                            )

            with A.MPMD(name="qkv_proj_comm_q_all_gather", type="communication") as CORE_ID:
                with A.Kernel(core_list=list(range(core_num))):
                    q_all_gather_line_pos = CORE_ID % core_array_size
                    owned_chunk_buffer = A.alloc(shape=(row_num, q_all_gather_chunk_length), dtype=dtype)
                    recv_from_neg = A.alloc(shape=(row_num, q_all_gather_chunk_length), dtype=dtype)
                    recv_from_pos = A.alloc(shape=(row_num, q_all_gather_chunk_length), dtype=dtype)

                    for t in A.Serial(core_array_size - 1):
                        if t == 0:
                            _copy_line_chunk(
                                input_attention,
                                owned_chunk_buffer,
                                row_num=row_num,
                                chunk_length=q_all_gather_chunk_length,
                                chunk_id=q_all_gather_line_pos,
                            )
                            if q_all_gather_line_pos > 0:
                                A.send(CORE_ID, _line_neighbor(core_array_size, 1, CORE_ID, -1), owned_chunk_buffer)
                            if q_all_gather_line_pos < core_array_size - 1:
                                A.send(CORE_ID, _line_neighbor(core_array_size, 1, CORE_ID, +1), owned_chunk_buffer)

                        if q_all_gather_line_pos < core_array_size - 1 and t > 0 and t <= q_all_gather_line_pos:
                            A.send(CORE_ID, _line_neighbor(core_array_size, 1, CORE_ID, +1), recv_from_neg)
                        if q_all_gather_line_pos > 0 and t > 0 and t <= core_array_size - 1 - q_all_gather_line_pos:
                            A.send(CORE_ID, _line_neighbor(core_array_size, 1, CORE_ID, -1), recv_from_pos)

                        if q_all_gather_line_pos > 0 and t < q_all_gather_line_pos:
                            A.recv(_line_neighbor(core_array_size, 1, CORE_ID, -1), CORE_ID, recv_from_neg)
                            _store_line_chunk(
                                recv_from_neg,
                                input_attention,
                                row_num=row_num,
                                chunk_length=q_all_gather_chunk_length,
                                chunk_id=q_all_gather_line_pos - 1 - t,
                            )
                        if q_all_gather_line_pos < core_array_size - 1 and t < core_array_size - 1 - q_all_gather_line_pos:
                            A.recv(_line_neighbor(core_array_size, 1, CORE_ID, +1), CORE_ID, recv_from_pos)
                            _store_line_chunk(
                                recv_from_pos,
                                input_attention,
                                row_num=row_num,
                                chunk_length=q_all_gather_chunk_length,
                                chunk_id=q_all_gather_line_pos + 1 + t,
                            )

            with A.MPMD(name="qkv_proj_comm_kv_reduce_then_scatter", type="communication") as CORE_ID:
                with A.Kernel(core_list=[i for i in range(core_array_size)]):
                    send_buffer = A.alloc(shape=(row_num, kv_length), dtype=dtype)
                    recv_buffer = A.alloc(shape=(x_scatter_first_half_per_hop_req_num[0], kv_length), dtype=dtype)
                    scatter_rows = x_scatter_first_half_per_hop_req_num[0]

                    # Stage #1: reduce, second MST half firsA.
                    for _ in A.Serial(1):
                        _copy_line_chunk(
                            output_qkv_proj,
                            send_buffer,
                            row_num=row_num,
                            chunk_length=kv_length,
                            chunk_id=0,
                            base_offset=q_length,
                        )
                        A.send(
                            CORE_ID,
                            CORE_ID + core_array_size,
                            send_buffer[
                                x_reduce_first_half_req_num: x_reduce_first_half_req_num + x_reduce_second_half_req_num,
                                :,
                            ],
                        )

                    # Stage #1: reduce, then forward the first MST half.
                    for _ in A.Serial(1):
                        A.send(
                            CORE_ID, CORE_ID + core_array_size, send_buffer[: x_reduce_first_half_req_num, :])

                    # Stage #2: scatter the reduced first-half payload back to the edge.
                    for _ in A.Serial(1):
                        A.recv(CORE_ID + core_array_size, CORE_ID, recv_buffer)
                        _store_line_chunk(
                            recv_buffer,
                            output_qkv_proj,
                            row_num=scatter_rows,
                            chunk_length=kv_length,
                            chunk_id=0,
                            base_offset=q_length,
                        )

                with A.Kernel(core_list=[i+core_array_size for i in range(core_array_size)]):
                    partial_sum_buffer = A.alloc(shape=(row_num, kv_length), dtype=dtype)
                    send_buffer1 = A.alloc(shape=(x_reduce_second_half_req_num, kv_length), dtype=dtype)
                    recv_buffer1 = A.alloc(shape=(x_reduce_first_half_req_num, kv_length), dtype=dtype)
                    recv_buffer2 = A.alloc(shape=(x_reduce_first_half_req_num, kv_length), dtype=dtype)
                    send_buffer2 = A.alloc(shape=(x_reduce_first_half_req_num, kv_length), dtype=dtype)
                    scatter_rows = x_scatter_first_half_per_hop_req_num[0]
                    retained_rows = x_reduce_first_half_req_num - scatter_rows

                    # Stage #1: reduce the second MST half toward the center.
                    for _ in A.Serial(1):
                        _copy_line_chunk(
                            output_qkv_proj,
                            partial_sum_buffer,
                            row_num=row_num,
                            chunk_length=kv_length,
                            chunk_id=0,
                            base_offset=q_length,
                        )
                        A.recv(CORE_ID - core_array_size, CORE_ID, send_buffer1)
                        A.add(
                            send_buffer1,
                            partial_sum_buffer[
                                x_reduce_first_half_req_num: x_reduce_first_half_req_num + x_reduce_second_half_req_num,
                                :,
                            ],
                            send_buffer1,
                        )

                    # Stage #1: merge both directions for the first MST half.
                    for _ in A.Serial(1):
                        A.send(CORE_ID, CORE_ID + core_array_size, send_buffer1)
                        A.recv(CORE_ID - core_array_size, CORE_ID, recv_buffer1)
                        A.recv(CORE_ID + core_array_size, CORE_ID, recv_buffer2)
                        A.add(recv_buffer1, recv_buffer2, send_buffer2)
                        A.add(send_buffer2, partial_sum_buffer[: x_reduce_first_half_req_num, :], send_buffer2)
                        _store_line_chunk(
                            send_buffer2[scatter_rows:, :],
                            output_qkv_proj,
                            row_num=retained_rows,
                            chunk_length=kv_length,
                            chunk_id=0,
                            base_offset=q_length,
                        )

                    # Stage #2: scatter the reduced prefix back to the upper half.
                    for _ in A.Serial(1):
                        A.send(
                            CORE_ID,
                            CORE_ID - core_array_size,
                            send_buffer2[: scatter_rows, :],
                        )

                with A.Kernel(core_list=[i+2*core_array_size for i in range(core_array_size)]):
                    partial_sum_buffer = A.alloc(shape=(row_num, kv_length), dtype=dtype)
                    send_buffer1 = A.alloc(shape=(x_reduce_first_half_req_num, kv_length), dtype=dtype)
                    recv_buffer1 = A.alloc(shape=(x_reduce_second_half_req_num, kv_length), dtype=dtype)
                    recv_buffer2 = A.alloc(shape=(x_reduce_second_half_req_num, kv_length), dtype=dtype)
                    send_buffer2 = A.alloc(shape=(x_reduce_second_half_req_num, kv_length), dtype=dtype)
                    scatter_rows = x_scatter_second_half_per_hop_req_num[0]
                    retained_rows = x_reduce_second_half_req_num - scatter_rows

                    # Stage #1: reduce the first MST half toward the center.
                    for _ in A.Serial(1):
                        _copy_line_chunk(
                            output_qkv_proj,
                            partial_sum_buffer,
                            row_num=row_num,
                            chunk_length=kv_length,
                            chunk_id=0,
                            base_offset=q_length,
                        )
                        A.recv(CORE_ID + core_array_size, CORE_ID, send_buffer1)
                        A.add(send_buffer1, partial_sum_buffer[: x_reduce_first_half_req_num, :], send_buffer1)

                    # Stage #1: merge both directions for the second MST half.
                    for _ in A.Serial(1):
                        A.send(CORE_ID, CORE_ID - core_array_size, send_buffer1)
                        A.recv(CORE_ID + core_array_size, CORE_ID, recv_buffer1)
                        A.recv(CORE_ID - core_array_size, CORE_ID, recv_buffer2)
                        A.add(recv_buffer1, recv_buffer2, send_buffer2)
                        A.add(
                            send_buffer2,
                            partial_sum_buffer[
                                x_reduce_first_half_req_num: x_reduce_first_half_req_num + x_reduce_second_half_req_num,
                                :,
                            ],
                            send_buffer2,
                        )
                        _store_line_chunk(
                            send_buffer2[: retained_rows, :],
                            output_qkv_proj,
                            row_num=retained_rows,
                            chunk_length=kv_length,
                            chunk_id=0,
                            base_offset=q_length,
                        )

                    # Stage #2: scatter the reduced suffix back to the lower half.
                    for _ in A.Serial(1):
                        A.send(
                            CORE_ID,
                            CORE_ID + core_array_size,
                            send_buffer2[retained_rows:, :],
                        )

                with A.Kernel(core_list=[i+3*core_array_size for i in range(core_array_size)]):
                    send_buffer = A.alloc(shape=(row_num, kv_length), dtype=dtype)
                    recv_buffer = A.alloc(shape=(x_scatter_second_half_per_hop_req_num[0], kv_length), dtype=dtype)
                    scatter_rows = x_scatter_second_half_per_hop_req_num[0]

                    # Stage #1: reduce, first MST half firsA.
                    for _ in A.Serial(1):
                        _copy_line_chunk(
                            output_qkv_proj,
                            send_buffer,
                            row_num=row_num,
                            chunk_length=kv_length,
                            chunk_id=0,
                            base_offset=q_length,
                        )
                        A.send(CORE_ID, CORE_ID - core_array_size, send_buffer[: x_reduce_first_half_req_num, :])

                    # Stage #1: reduce, then forward the second MST half.
                    for _ in A.Serial(1):
                        A.send(
                            CORE_ID,
                            CORE_ID - core_array_size,
                            send_buffer[
                                x_reduce_first_half_req_num: x_reduce_first_half_req_num + x_reduce_second_half_req_num,
                                :,
                            ],
                        )

                    # Stage #2: scatter the reduced second-half payload back to the edge.
                    for _ in A.Serial(1):
                        A.recv(CORE_ID - core_array_size, CORE_ID, recv_buffer)
                        _store_line_chunk(
                            recv_buffer,
                            output_qkv_proj,
                            row_num=scatter_rows,
                            chunk_length=kv_length,
                            chunk_id=0,
                            base_offset=q_length,
                        )

            with A.MPMD(name="qkv_proj_comm_kv_gather_then_scatter", type="communication") as CORE_ID:
                with A.Kernel(core_list=[i*core_array_size for i in range(core_array_size)]):
                    line_row_0 = CORE_ID // core_array_size
                    line_pos_0 = CORE_ID % core_array_size
                    gather_first_req_num_0 = y_gather_first_half_req_num[line_row_0]
                    gather_second_req_num_0 = y_gather_second_half_req_num[line_row_0]
                    gather_row_num_0 = gather_first_req_num_0 + gather_second_req_num_0
                    scatter_row_num_0 = y_scatter_first_half_per_hop_req_num[line_row_0][0]
                    slot_mapping_0 = last_slot_mapping[line_row_0][line_pos_0]
                    send_buffer = A.alloc(shape=(gather_row_num_0, kv_length), dtype=dtype)
                    recv_buffer = A.alloc(shape=(scatter_row_num_0, kv_length*core_array_size), dtype=dtype)

                    # Stage #1: gather, second MST payload firsA.
                    for _ in A.Serial(1):
                        _copy_line_chunk(
                            output_qkv_proj,
                            send_buffer,
                            row_num=gather_row_num_0,
                            chunk_length=kv_length,
                            chunk_id=0,
                            base_offset=q_length,
                        )
                        A.send(CORE_ID,
                            CORE_ID + 1,
                            send_buffer[
                                gather_first_req_num_0: gather_row_num_0,
                                :,
                            ],
                        )

                    # Stage #1: gather, then forward the first MST payload.
                    for _ in A.Serial(1):
                        A.send(CORE_ID, CORE_ID + 1, send_buffer[: gather_first_req_num_0, :])

                    # Stage #2: scatter the gathered vectors back to the left edge.
                    for _ in A.Serial(1):
                        A.recv(CORE_ID + 1, CORE_ID, recv_buffer)
                        for h in range(num_kv_heads):
                            for s in range(len(slot_mapping_0)):
                                _store_kv_cache_vector(
                                    recv_buffer,
                                    kv_cache,
                                    src_row=s,
                                    head_idx=h,
                                    slot_id=slot_mapping_0[s],
                                    head_dim=head_dim,
                                    total_slot_num=total_slot_num,
                                )

                with A.Kernel(core_list=[i*core_array_size+1 for i in range(core_array_size)]):
                    line_row_1 = CORE_ID // core_array_size
                    line_pos_1 = CORE_ID % core_array_size
                    gather_first_req_num_1 = y_gather_first_half_req_num[line_row_1]
                    gather_second_req_num_1 = y_gather_second_half_req_num[line_row_1]
                    gather_row_num_1 = gather_first_req_num_1 + gather_second_req_num_1
                    scatter_row_num_1 = y_scatter_first_half_per_hop_req_num[line_row_1][0]
                    slot_mapping_1 = last_slot_mapping[line_row_1][line_pos_1]
                    partial_vector_buffer = A.alloc(shape=(gather_row_num_1, kv_length), dtype=dtype)
                    send_buffer1 = A.alloc(shape=(gather_second_req_num_1, 2*kv_length), dtype=dtype)
                    send_buffer2 = A.alloc(shape=(gather_first_req_num_1, core_array_size*kv_length), dtype=dtype)

                    # Stage #1: gather the second MST half from the left edge.
                    for _ in A.Serial(1):
                        _copy_line_chunk(
                            output_qkv_proj,
                            partial_vector_buffer,
                            row_num=gather_row_num_1,
                            chunk_length=kv_length,
                            chunk_id=0,
                            base_offset=q_length,
                        )
                        A.copy(partial_vector_buffer[: gather_first_req_num_1, :], send_buffer2[:, kv_length: 2*kv_length])
                        A.recv(CORE_ID - 1, CORE_ID, send_buffer1[:, : kv_length])

                    # Stage #1: merge both directions for the first MST half.
                    for _ in A.Serial(1):
                        A.send(CORE_ID, CORE_ID + 1, send_buffer1)
                        A.recv(CORE_ID - 1, CORE_ID, send_buffer2[:, : kv_length])
                        A.recv(CORE_ID + 1, CORE_ID, send_buffer2[:, 2*kv_length: ])
                        for h in range(num_kv_heads):
                            for s in range(len(slot_mapping_1)):
                                _store_kv_cache_vector(
                                    send_buffer2,
                                    kv_cache,
                                    src_row=scatter_row_num_1 + s,
                                    head_idx=h,
                                    slot_id=slot_mapping_1[s],
                                    head_dim=head_dim,
                                    total_slot_num=total_slot_num,
                                )

                    # Stage #2: scatter the first-half gathered payload back to the left edge.
                    for _ in A.Serial(1):
                        A.send(CORE_ID, CORE_ID - 1, send_buffer2[: scatter_row_num_1, :])

                with A.Kernel(core_list=[i*core_array_size+2 for i in range(core_array_size)]):
                    line_row_2 = CORE_ID // core_array_size
                    line_pos_2 = CORE_ID % core_array_size
                    gather_first_req_num_2 = y_gather_first_half_req_num[line_row_2]
                    gather_second_req_num_2 = y_gather_second_half_req_num[line_row_2]
                    gather_row_num_2 = gather_first_req_num_2 + gather_second_req_num_2
                    scatter_row_num_2 = y_scatter_first_half_per_hop_req_num[line_row_2][0]
                    slot_mapping_2 = last_slot_mapping[line_row_2][line_pos_2]
                    partial_vector_buffer = A.alloc(shape=(gather_row_num_2, kv_length), dtype=dtype)
                    send_buffer1 = A.alloc(shape=(gather_first_req_num_2, 2*kv_length), dtype=dtype)
                    send_buffer2 = A.alloc(shape=(gather_second_req_num_2, core_array_size*kv_length), dtype=dtype)

                    # Stage #1: gather the first MST half from the right edge.
                    for _ in A.Serial(1):
                        _copy_line_chunk(
                            output_qkv_proj,
                            partial_vector_buffer,
                            row_num=gather_row_num_2,
                            chunk_length=kv_length,
                            chunk_id=0,
                            base_offset=q_length,
                        )
                        A.copy(
                            partial_vector_buffer[gather_first_req_num_2: gather_row_num_2, :],
                            send_buffer2[:, 2*kv_length: 3*kv_length],
                        )
                        A.recv(CORE_ID + 1, CORE_ID, send_buffer1[:, kv_length: 2*kv_length])

                    # Stage #1: merge both directions for the second MST half.
                    for _ in A.Serial(1):
                        A.send(CORE_ID, CORE_ID - 1, send_buffer1)
                        A.recv(CORE_ID + 1, CORE_ID, send_buffer2[:, 3*kv_length: ])
                        A.recv(CORE_ID - 1, CORE_ID, send_buffer2[:, : 2*kv_length])
                        for h in range(num_kv_heads):
                            for s in range(len(slot_mapping_2)):
                                _store_kv_cache_vector(
                                    send_buffer2,
                                    kv_cache,
                                    src_row=s,
                                    head_idx=h,
                                    slot_id=slot_mapping_2[s],
                                    head_dim=head_dim,
                                    total_slot_num=total_slot_num,
                                )

                    # Stage #2: scatter the second-half gathered payload back to the right edge.
                    for _ in A.Serial(1):
                        A.send(
                            CORE_ID,
                            CORE_ID + 1,
                            send_buffer2[gather_second_req_num_2 - scatter_row_num_2:, :],
                        )

                with A.Kernel(core_list=[i*core_array_size+3 for i in range(core_array_size)]):
                    line_row_3 = CORE_ID // core_array_size
                    line_pos_3 = CORE_ID % core_array_size
                    gather_first_req_num_3 = y_gather_first_half_req_num[line_row_3]
                    gather_second_req_num_3 = y_gather_second_half_req_num[line_row_3]
                    gather_row_num_3 = gather_first_req_num_3 + gather_second_req_num_3
                    scatter_row_num_3 = y_scatter_second_half_per_hop_req_num[line_row_3][0]
                    slot_mapping_3 = last_slot_mapping[line_row_3][line_pos_3]
                    send_buffer = A.alloc(shape=(gather_row_num_3, kv_length), dtype=dtype)
                    recv_buffer = A.alloc(shape=(scatter_row_num_3, kv_length*core_array_size), dtype=dtype)

                    # Stage #1: gather, first MST payload firsA.
                    for _ in A.Serial(1):
                        _copy_line_chunk(
                            output_qkv_proj,
                            send_buffer,
                            row_num=gather_row_num_3,
                            chunk_length=kv_length,
                            chunk_id=0,
                            base_offset=q_length,
                        )
                        A.send(CORE_ID, CORE_ID - 1, send_buffer[: gather_first_req_num_3, :])

                    # Stage #1: gather, then forward the second MST payload.
                    for _ in A.Serial(1):
                        A.send(
                            CORE_ID,
                            CORE_ID - 1,
                            send_buffer[gather_first_req_num_3: gather_row_num_3, :],
                        )

                    # Stage #2: scatter the gathered vectors back to the right edge.
                    for _ in A.Serial(1):
                        A.recv(CORE_ID - 1, CORE_ID, recv_buffer)
                        for h in range(num_kv_heads):
                            for s in range(len(slot_mapping_3)):
                                _store_kv_cache_vector(
                                    recv_buffer,
                                    kv_cache,
                                    src_row=s,
                                    head_idx=h,
                                    slot_id=slot_mapping_3[s],
                                    head_dim=head_dim,
                                    total_slot_num=total_slot_num,
                                )

            # core_R: request num (i.e., batch size)
            # core_KV: KV head num
            # core_N: KV group num (Q head num per KV head)
            # core_S: context length
            # core_H: head dim
            with A.SPMD(name="attention", type="decode-attention", **attention_kwargs) as (core_R, core_KV, core_N, core_S, core_H):
                with A.Kernel(A.ceildiv(core_S, min_tS), autotune=True) as (bS):
                    input_attention = A.Tensor(shape=(core_R, core_KV * core_N * core_H), strides=(core_KV * core_N * core_H, 1), dtype=dtype)
                    kv_cache = A.Tensor(shape=(core_KV*core_S, 2*core_H), strides=(2*core_H, 1), dtype=dtype)
                    output_attention = A.Tensor(shape=(core_R, core_KV * core_N * core_H), strides=(core_KV * core_N * core_H, 1), dtype=dtype)

                    attn_input_tile = A.alloc(shape=(core_N, core_H), dtype=dtype)
                    kv_cache_tile = A.alloc(shape=(min_tS, 2*core_H), dtype=dtype)
                    acc_tile = A.alloc(shape=(core_N, min_tS), dtype=dtype)
                    scores_max_tile = A.alloc(shape=(core_N,), dtype=dtype)
                    scores_max_prev_tile = A.alloc(shape=(core_N,), dtype=dtype)
                    scores_scale_tile = A.alloc(shape=(core_N,), dtype=dtype)
                    log_sum_prev_tile = A.alloc(shape=(core_N,), dtype=dtype)
                    log_sum_tile = A.alloc(shape=(core_N,), dtype=dtype)
                    attn_output_tile_new = A.alloc(shape=(core_N, core_H), dtype=dtype)
                    attn_output_tile = A.alloc(shape=(core_N, core_H), dtype=dtype)

                    for r in A.Serial(core_R):
                        for kv in A.Serial(core_KV):
                            A.copy(input_attention[r, kv * core_N * core_H: (kv + 1) * core_N * core_H], attn_input_tile)
                            A.fill(scores_max_tile, -A.infinity(dtype))
                            A.clear(log_sum_tile)
                            A.clear(attn_output_tile)
                            for s in A.Serial(bS):
                                A.copy(kv_cache[kv*core_S + s*min_tS: kv*core_S + (s+1)*min_tS, :], kv_cache_tile)
                                A.copy(scores_max_tile, scores_max_prev_tile)
                                # 1st GEMM
                                A.gemm(attn_input_tile, kv_cache_tile[:,:core_H], acc_tile, transpose_B=True)
                                # online softmax
                                # get m_i
                                A.reduce_max(acc_tile, scores_max_tile, dim=1, clear=True)
                                A.max(scores_max_tile, scores_max_prev_tile, scores_max_tile)
                                # get e^{x_i - m_i}
                                A.sub(acc_tile, scores_max_tile, acc_tile)
                                A.exp(acc_tile, acc_tile)
                                # get l_i = l_{i-1}*e^{m_{i-1}-m_i} + sum(e^{x_i - m_i})
                                A.sub(scores_max_prev_tile, scores_max_tile, scores_scale_tile)
                                A.exp(scores_scale_tile, scores_scale_tile)
                                A.mul(log_sum_tile, scores_scale_tile, log_sum_prev_tile)
                                A.reduce_sum(acc_tile, log_sum_tile, dim=1, clear=True)
                                A.add(log_sum_tile, log_sum_prev_tile, log_sum_tile)
                                # 2nd GEMM
                                A.gemm(acc_tile, kv_cache_tile[:,core_H:], attn_output_tile_new)
                                # Partial sum acclumulation
                                A.mul(attn_output_tile, log_sum_prev_tile, attn_output_tile) # get O_{i-1}*e^{m_{i-1}-m_i}
                                A.add(attn_output_tile, attn_output_tile_new, attn_output_tile) # get O_{i-1}*e^{m_{i-1}-m_i} + O_new
                                A.div(attn_output_tile, log_sum_tile, attn_output_tile) # get O_i = (O_{i-1}*e^{m_{i-1}-m_i} + O_new) / l_i
                            A.copy(attn_output_tile, output_attention[r, kv * core_N * core_H: (kv + 1) * core_N * core_H])

            row_num = attention_comm_row_num
            with A.MPMD(name="attention_comm_attn_reduce_scatter", type="communication") as CORE_ID:
                # Here we use 2D ring reduce scatter, so all cores share the same program
                with A.Kernel(core_list=list(range(core_num))):
                    ring_rank = core_rank_map[CORE_ID]
                    next_core = rank_core_map[(ring_rank + 1) % core_num]
                    prev_core = rank_core_map[(ring_rank - 1 + core_num) % core_num]
                    send_buffer = A.alloc(shape=(row_num, attn_reduce_scatter_chunk_length), dtype=dtype)
                    recv_buffer = A.alloc(shape=(row_num, attn_reduce_scatter_chunk_length), dtype=dtype)
                    recv_reduce_buffer = A.alloc(shape=(row_num, attn_reduce_scatter_chunk_length), dtype=dtype)

                    for t in A.Serial(core_num - 1):
                        current_chunk_id = rank_core_map[(core_num - 1 + ring_rank - t) % core_num]
                        next_chunk_id = rank_core_map[(core_num - 2 + ring_rank - t) % core_num]
                        if t % 2 == 0:
                            if t == 0:
                                _copy_line_chunk(
                                    output_attention,
                                    send_buffer,
                                    row_num=row_num,
                                    chunk_length=attn_reduce_scatter_chunk_length,
                                    chunk_id=current_chunk_id,
                                )
                            A.send(CORE_ID, next_core, send_buffer)

                            _copy_line_chunk(
                                output_attention,
                                recv_reduce_buffer,
                                row_num=row_num,
                                chunk_length=attn_reduce_scatter_chunk_length,
                                chunk_id=next_chunk_id,
                            )
                            A.recv(prev_core, CORE_ID, recv_buffer)
                            A.add(recv_reduce_buffer, recv_buffer, recv_buffer)

                            if t == core_num - 2:
                                _store_line_chunk(
                                    recv_buffer,
                                    input_o_proj,
                                    row_num=row_num,
                                    chunk_length=attn_reduce_scatter_chunk_length,
                                    chunk_id=CORE_ID % core_array_size,
                                )
                        else:
                            A.send(CORE_ID, next_core, recv_buffer)

                            _copy_line_chunk(
                                output_attention,
                                recv_reduce_buffer,
                                row_num=row_num,
                                chunk_length=attn_reduce_scatter_chunk_length,
                                chunk_id=next_chunk_id,
                            )
                            A.recv(prev_core, CORE_ID, send_buffer)
                            A.add(recv_reduce_buffer, send_buffer, send_buffer)

                            if t == core_num - 2:
                                _store_line_chunk(
                                    send_buffer,
                                    input_o_proj,
                                    row_num=row_num,
                                    chunk_length=attn_reduce_scatter_chunk_length,
                                    chunk_id=CORE_ID % core_array_size,
                                )
                    
            with A.MPMD(name="attention_comm_attn_all_gather", type="communication") as CORE_ID:
                with A.Kernel(core_list=list(range(core_num))):
                    attn_all_gather_line_pos = CORE_ID % core_array_size
                    owned_chunk_buffer = A.alloc(shape=(row_num, attn_all_gather_chunk_length), dtype=dtype)
                    recv_from_neg = A.alloc(shape=(row_num, attn_all_gather_chunk_length), dtype=dtype)
                    recv_from_pos = A.alloc(shape=(row_num, attn_all_gather_chunk_length), dtype=dtype)

                    for t in A.Serial(core_array_size - 1):
                        if t == 0:
                            _copy_line_chunk(
                                input_o_proj,
                                owned_chunk_buffer,
                                row_num=row_num,
                                chunk_length=attn_all_gather_chunk_length,
                                chunk_id=attn_all_gather_line_pos,
                            )
                            if attn_all_gather_line_pos > 0:
                                A.send(CORE_ID, _line_neighbor(core_array_size, 1, CORE_ID, -1), owned_chunk_buffer)
                            if attn_all_gather_line_pos < core_array_size - 1:
                                A.send(CORE_ID, _line_neighbor(core_array_size, 1, CORE_ID, +1), owned_chunk_buffer)

                        if attn_all_gather_line_pos < core_array_size - 1 and t > 0 and t <= attn_all_gather_line_pos:
                            A.send(CORE_ID, _line_neighbor(core_array_size, 1, CORE_ID, +1), recv_from_neg)
                        if attn_all_gather_line_pos > 0 and t > 0 and t <= core_array_size - 1 - attn_all_gather_line_pos:
                            A.send(CORE_ID, _line_neighbor(core_array_size, 1, CORE_ID, -1), recv_from_pos)

                        if attn_all_gather_line_pos > 0 and t < attn_all_gather_line_pos:
                            A.recv(_line_neighbor(core_array_size, 1, CORE_ID, -1), CORE_ID, recv_from_neg)
                            _store_line_chunk(
                                recv_from_neg,
                                input_o_proj,
                                row_num=row_num,
                                chunk_length=attn_all_gather_chunk_length,
                                chunk_id=attn_all_gather_line_pos - 1 - t,
                            )
                        if attn_all_gather_line_pos < core_array_size - 1 and t < core_array_size - 1 - attn_all_gather_line_pos:
                            A.recv(_line_neighbor(core_array_size, 1, CORE_ID, +1), CORE_ID, recv_from_pos)
                            _store_line_chunk(
                                recv_from_pos,
                                input_o_proj,
                                row_num=row_num,
                                chunk_length=attn_all_gather_chunk_length,
                                chunk_id=attn_all_gather_line_pos + 1 + t,
                            )

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

            row_num = o_proj_comm_row_num
            with A.MPMD(name="o_proj_comm_1d_all_reduce", type="communication") as CORE_ID:
                with A.Kernel(core_list=list(range(core_num))):
                    o_proj_reduce_line_pos = CORE_ID // core_array_size
                    pos_wave_buffer = A.alloc(shape=(row_num, o_all_reduce_chunk_length), dtype=dtype)
                    neg_wave_buffer = A.alloc(shape=(row_num, o_all_reduce_chunk_length), dtype=dtype)
                    recv_from_neg = A.alloc(shape=(row_num, o_all_reduce_chunk_length), dtype=dtype)
                    recv_from_pos = A.alloc(shape=(row_num, o_all_reduce_chunk_length), dtype=dtype)
                    owned_chunk_buffer = A.alloc(shape=(row_num, o_all_reduce_chunk_length), dtype=dtype)

                    for t in A.Serial(core_array_size - 1):
                        if o_proj_reduce_line_pos == 0:
                            _copy_line_chunk(
                                output_o_proj,
                                pos_wave_buffer,
                                row_num=row_num,
                                chunk_length=o_all_reduce_chunk_length,
                                chunk_id=t,
                            )
                            A.send(CORE_ID, _line_neighbor(core_array_size, 0, CORE_ID, +1), pos_wave_buffer)
                        if 0 < o_proj_reduce_line_pos < core_array_size - 1 and o_proj_reduce_line_pos <= t:
                            A.send(CORE_ID, _line_neighbor(core_array_size, 0, CORE_ID, +1), pos_wave_buffer)
                        if o_proj_reduce_line_pos > 0 and o_proj_reduce_line_pos - 1 <= t:
                            _copy_line_chunk(
                                output_o_proj,
                                pos_wave_buffer,
                                row_num=row_num,
                                chunk_length=o_all_reduce_chunk_length,
                                chunk_id=t - (o_proj_reduce_line_pos - 1),
                            )
                            A.recv(_line_neighbor(core_array_size, 0, CORE_ID, -1), CORE_ID, recv_from_neg)
                            A.add(pos_wave_buffer, recv_from_neg, pos_wave_buffer)

                        if o_proj_reduce_line_pos == core_array_size - 1:
                            _copy_line_chunk(
                                output_o_proj,
                                neg_wave_buffer,
                                row_num=row_num,
                                chunk_length=o_all_reduce_chunk_length,
                                chunk_id=core_array_size - 1 - t,
                            )
                            A.send(CORE_ID, _line_neighbor(core_array_size, 0, CORE_ID, -1), neg_wave_buffer)
                        if 0 < o_proj_reduce_line_pos < core_array_size - 1 and o_proj_reduce_line_pos >= core_array_size - 1 - t:
                            A.send(CORE_ID, _line_neighbor(core_array_size, 0, CORE_ID, -1), neg_wave_buffer)
                        if o_proj_reduce_line_pos < core_array_size - 1 and o_proj_reduce_line_pos + 1 >= core_array_size - 1 - t:
                            _copy_line_chunk(
                                output_o_proj,
                                neg_wave_buffer,
                                row_num=row_num,
                                chunk_length=o_all_reduce_chunk_length,
                                chunk_id=2 * core_array_size - 3 - o_proj_reduce_line_pos - t,
                            )
                            A.recv(_line_neighbor(core_array_size, 0, CORE_ID, +1), CORE_ID, recv_from_pos)
                            A.add(neg_wave_buffer, recv_from_pos, neg_wave_buffer)
                        if t == core_array_size - 2:
                            if o_proj_reduce_line_pos == 0:
                                A.copy(neg_wave_buffer, owned_chunk_buffer)
                            elif o_proj_reduce_line_pos == core_array_size - 1:
                                A.copy(pos_wave_buffer, owned_chunk_buffer)
                            elif o_proj_reduce_line_pos < core_array_size // 2:
                                A.add(pos_wave_buffer, recv_from_pos, owned_chunk_buffer)
                            else:
                                A.add(recv_from_neg, neg_wave_buffer, owned_chunk_buffer)

                    for t in A.Serial(core_array_size - 1):
                        if t == 0:
                            _store_line_chunk(
                                owned_chunk_buffer,
                                input_down_proj,
                                row_num=row_num,
                                chunk_length=o_all_reduce_chunk_length,
                                chunk_id=core_array_size - 1 - o_proj_reduce_line_pos,
                            )
                            if o_proj_reduce_line_pos > 0:
                                A.send(CORE_ID, _line_neighbor(core_array_size, 0, CORE_ID, -1), owned_chunk_buffer)
                            if o_proj_reduce_line_pos < core_array_size - 1:
                                A.send(CORE_ID, _line_neighbor(core_array_size, 0, CORE_ID, +1), owned_chunk_buffer)

                        if o_proj_reduce_line_pos < core_array_size - 1 and t > 0 and t <= o_proj_reduce_line_pos:
                            A.send(CORE_ID, _line_neighbor(core_array_size, 0, CORE_ID, +1), recv_from_neg)
                        if o_proj_reduce_line_pos > 0 and t > 0 and t <= core_array_size - 1 - o_proj_reduce_line_pos:
                            A.send(CORE_ID, _line_neighbor(core_array_size, 0, CORE_ID, -1), recv_from_pos)

                        if o_proj_reduce_line_pos > 0 and t < o_proj_reduce_line_pos:
                            A.recv(_line_neighbor(core_array_size, 0, CORE_ID, -1), CORE_ID, recv_from_neg)
                            _store_line_chunk(
                                recv_from_neg,
                                input_down_proj,
                                row_num=row_num,
                                chunk_length=o_all_reduce_chunk_length,
                                chunk_id=core_array_size - o_proj_reduce_line_pos + t,
                            )
                        if o_proj_reduce_line_pos < core_array_size - 1 and t < core_array_size - 1 - o_proj_reduce_line_pos:
                            A.recv(_line_neighbor(core_array_size, 0, CORE_ID, +1), CORE_ID, recv_from_pos)
                            _store_line_chunk(
                                recv_from_pos,
                                input_down_proj,
                                row_num=row_num,
                                chunk_length=o_all_reduce_chunk_length,
                                chunk_id=core_array_size - o_proj_reduce_line_pos - 2 - t,
                            )

            # (1) inter-device all-reduce/all-to-all will be automatically induced by tiling explorer
            # (2) For simplicity, MoE's batched GEMM will also be automatically induced by tiling explorer, and we only describe one GEMM here
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
            
            # This is conducted along Y axis (1th axis), while other 1d_all_reduce are conducted along X axis (0th axis)
            row_num = down_proj_comm_row_num
            with A.MPMD(name="down_proj_comm_1d_all_reduce", type="communication") as CORE_ID:
                with A.Kernel(core_list=list(range(core_num))):
                    down_proj_reduce_line_pos = CORE_ID % core_array_size
                    pos_wave_buffer = A.alloc(shape=(row_num, down_all_reduce_chunk_length), dtype=dtype)
                    neg_wave_buffer = A.alloc(shape=(row_num, down_all_reduce_chunk_length), dtype=dtype)
                    recv_from_neg = A.alloc(shape=(row_num, down_all_reduce_chunk_length), dtype=dtype)
                    recv_from_pos = A.alloc(shape=(row_num, down_all_reduce_chunk_length), dtype=dtype)
                    owned_chunk_buffer = A.alloc(shape=(row_num, down_all_reduce_chunk_length), dtype=dtype)

                    for t in A.Serial(core_array_size - 1):
                        if down_proj_reduce_line_pos == 0:
                            _copy_line_chunk(
                                output_down_proj,
                                pos_wave_buffer,
                                row_num=row_num,
                                chunk_length=down_all_reduce_chunk_length,
                                chunk_id=t,
                            )
                            A.send(CORE_ID, _line_neighbor(core_array_size, 1, CORE_ID, +1), pos_wave_buffer)
                        if 0 < down_proj_reduce_line_pos < core_array_size - 1 and down_proj_reduce_line_pos <= t:
                            A.send(CORE_ID, _line_neighbor(core_array_size, 1, CORE_ID, +1), pos_wave_buffer)
                        if down_proj_reduce_line_pos > 0 and down_proj_reduce_line_pos - 1 <= t:
                            _copy_line_chunk(
                                output_down_proj,
                                pos_wave_buffer,
                                row_num=row_num,
                                chunk_length=down_all_reduce_chunk_length,
                                chunk_id=t - (down_proj_reduce_line_pos - 1),
                            )
                            A.recv(_line_neighbor(core_array_size, 1, CORE_ID, -1), CORE_ID, recv_from_neg)
                            A.add(pos_wave_buffer, recv_from_neg, pos_wave_buffer)

                        if down_proj_reduce_line_pos == core_array_size - 1:
                            _copy_line_chunk(
                                output_down_proj,
                                neg_wave_buffer,
                                row_num=row_num,
                                chunk_length=down_all_reduce_chunk_length,
                                chunk_id=core_array_size - 1 - t,
                            )
                            A.send(CORE_ID, _line_neighbor(core_array_size, 1, CORE_ID, -1), neg_wave_buffer)
                        if 0 < down_proj_reduce_line_pos < core_array_size - 1 and down_proj_reduce_line_pos >= core_array_size - 1 - t:
                            A.send(CORE_ID, _line_neighbor(core_array_size, 1, CORE_ID, -1), neg_wave_buffer)
                        if down_proj_reduce_line_pos < core_array_size - 1 and down_proj_reduce_line_pos + 1 >= core_array_size - 1 - t:
                            _copy_line_chunk(
                                output_down_proj,
                                neg_wave_buffer,
                                row_num=row_num,
                                chunk_length=down_all_reduce_chunk_length,
                                chunk_id=2 * core_array_size - 3 - down_proj_reduce_line_pos - t,
                            )
                            A.recv(_line_neighbor(core_array_size, 1, CORE_ID, +1), CORE_ID, recv_from_pos)
                            A.add(neg_wave_buffer, recv_from_pos, neg_wave_buffer)
                        if t == core_array_size - 2:
                            if down_proj_reduce_line_pos == 0:
                                A.copy(neg_wave_buffer, owned_chunk_buffer)
                            elif down_proj_reduce_line_pos == core_array_size - 1:
                                A.copy(pos_wave_buffer, owned_chunk_buffer)
                            elif down_proj_reduce_line_pos < core_array_size // 2:
                                A.add(pos_wave_buffer, recv_from_pos, owned_chunk_buffer)
                            else:
                                A.add(recv_from_neg, neg_wave_buffer, owned_chunk_buffer)

                    for t in A.Serial(core_array_size - 1):
                        if t == 0:
                            _store_line_chunk(
                                owned_chunk_buffer,
                                input_up_proj,
                                row_num=row_num,
                                chunk_length=down_all_reduce_chunk_length,
                                chunk_id=core_array_size - 1 - down_proj_reduce_line_pos,
                            )
                            if down_proj_reduce_line_pos > 0:
                                A.send(CORE_ID, _line_neighbor(core_array_size, 1, CORE_ID, -1), owned_chunk_buffer)
                            if down_proj_reduce_line_pos < core_array_size - 1:
                                A.send(CORE_ID, _line_neighbor(core_array_size, 1, CORE_ID, +1), owned_chunk_buffer)

                        if down_proj_reduce_line_pos < core_array_size - 1 and t > 0 and t <= down_proj_reduce_line_pos:
                            A.send(CORE_ID, _line_neighbor(core_array_size, 1, CORE_ID, +1), recv_from_neg)
                        if down_proj_reduce_line_pos > 0 and t > 0 and t <= core_array_size - 1 - down_proj_reduce_line_pos:
                            A.send(CORE_ID, _line_neighbor(core_array_size, 1, CORE_ID, -1), recv_from_pos)

                        if down_proj_reduce_line_pos > 0 and t < down_proj_reduce_line_pos:
                            A.recv(_line_neighbor(core_array_size, 1, CORE_ID, -1), CORE_ID, recv_from_neg)
                            _store_line_chunk(
                                recv_from_neg,
                                input_up_proj,
                                row_num=row_num,
                                chunk_length=down_all_reduce_chunk_length,
                                chunk_id=core_array_size - down_proj_reduce_line_pos + t,
                            )
                        if down_proj_reduce_line_pos < core_array_size - 1 and t < core_array_size - 1 - down_proj_reduce_line_pos:
                            A.recv(_line_neighbor(core_array_size, 1, CORE_ID, +1), CORE_ID, recv_from_pos)
                            _store_line_chunk(
                                recv_from_pos,
                                input_up_proj,
                                row_num=row_num,
                                chunk_length=down_all_reduce_chunk_length,
                                chunk_id=core_array_size - down_proj_reduce_line_pos - 2 - t,
                            )

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

            row_num = up_proj_comm_row_num
            with A.MPMD(name="up_proj_comm_1d_all_reduce", type="communication") as CORE_ID:
                with A.Kernel(core_list=list(range(core_num))):
                    up_proj_reduce_line_pos = CORE_ID // core_array_size
                    pos_wave_buffer = A.alloc(shape=(row_num, up_all_reduce_chunk_length), dtype=dtype)
                    neg_wave_buffer = A.alloc(shape=(row_num, up_all_reduce_chunk_length), dtype=dtype)
                    recv_from_neg = A.alloc(shape=(row_num, up_all_reduce_chunk_length), dtype=dtype)
                    recv_from_pos = A.alloc(shape=(row_num, up_all_reduce_chunk_length), dtype=dtype)
                    owned_chunk_buffer = A.alloc(shape=(row_num, up_all_reduce_chunk_length), dtype=dtype)

                    for t in A.Serial(core_array_size - 1):
                        if up_proj_reduce_line_pos == 0:
                            _copy_line_chunk(
                                output_up_proj,
                                pos_wave_buffer,
                                row_num=row_num,
                                chunk_length=up_all_reduce_chunk_length,
                                chunk_id=t,
                            )
                            A.send(CORE_ID, _line_neighbor(core_array_size, 0, CORE_ID, +1), pos_wave_buffer)
                        if 0 < up_proj_reduce_line_pos < core_array_size - 1 and up_proj_reduce_line_pos <= t:
                            A.send(CORE_ID, _line_neighbor(core_array_size, 0, CORE_ID, +1), pos_wave_buffer)
                        if up_proj_reduce_line_pos > 0 and up_proj_reduce_line_pos - 1 <= t:
                            _copy_line_chunk(
                                output_up_proj,
                                pos_wave_buffer,
                                row_num=row_num,
                                chunk_length=up_all_reduce_chunk_length,
                                chunk_id=t - (up_proj_reduce_line_pos - 1),
                            )
                            A.recv(_line_neighbor(core_array_size, 0, CORE_ID, -1), CORE_ID, recv_from_neg)
                            A.add(pos_wave_buffer, recv_from_neg, pos_wave_buffer)

                        if up_proj_reduce_line_pos == core_array_size - 1:
                            _copy_line_chunk(
                                output_up_proj,
                                neg_wave_buffer,
                                row_num=row_num,
                                chunk_length=up_all_reduce_chunk_length,
                                chunk_id=core_array_size - 1 - t,
                            )
                            A.send(CORE_ID, _line_neighbor(core_array_size, 0, CORE_ID, -1), neg_wave_buffer)
                        if 0 < up_proj_reduce_line_pos < core_array_size - 1 and up_proj_reduce_line_pos >= core_array_size - 1 - t:
                            A.send(CORE_ID, _line_neighbor(core_array_size, 0, CORE_ID, -1), neg_wave_buffer)
                        if up_proj_reduce_line_pos < core_array_size - 1 and up_proj_reduce_line_pos + 1 >= core_array_size - 1 - t:
                            _copy_line_chunk(
                                output_up_proj,
                                neg_wave_buffer,
                                row_num=row_num,
                                chunk_length=up_all_reduce_chunk_length,
                                chunk_id=2 * core_array_size - 3 - up_proj_reduce_line_pos - t,
                            )
                            A.recv(_line_neighbor(core_array_size, 0, CORE_ID, +1), CORE_ID, recv_from_pos)
                            A.add(neg_wave_buffer, recv_from_pos, neg_wave_buffer)
                        if t == core_array_size - 2:
                            if up_proj_reduce_line_pos == 0:
                                A.copy(neg_wave_buffer, owned_chunk_buffer)
                            elif up_proj_reduce_line_pos == core_array_size - 1:
                                A.copy(pos_wave_buffer, owned_chunk_buffer)
                            elif up_proj_reduce_line_pos < core_array_size // 2:
                                A.add(pos_wave_buffer, recv_from_pos, owned_chunk_buffer)
                            else:
                                A.add(recv_from_neg, neg_wave_buffer, owned_chunk_buffer)

                    for t in A.Serial(core_array_size - 1):
                        if t == 0:
                            _store_line_chunk(
                                owned_chunk_buffer,
                                input_qkv_proj,
                                row_num=row_num,
                                chunk_length=up_all_reduce_chunk_length,
                                chunk_id=core_array_size - 1 - up_proj_reduce_line_pos,
                            )
                            if up_proj_reduce_line_pos > 0:
                                A.send(CORE_ID, _line_neighbor(core_array_size, 0, CORE_ID, -1), owned_chunk_buffer)
                            if up_proj_reduce_line_pos < core_array_size - 1:
                                A.send(CORE_ID, _line_neighbor(core_array_size, 0, CORE_ID, +1), owned_chunk_buffer)

                        if up_proj_reduce_line_pos < core_array_size - 1 and t > 0 and t <= up_proj_reduce_line_pos:
                            A.send(CORE_ID, _line_neighbor(core_array_size, 0, CORE_ID, +1), recv_from_neg)
                        if up_proj_reduce_line_pos > 0 and t > 0 and t <= core_array_size - 1 - up_proj_reduce_line_pos:
                            A.send(CORE_ID, _line_neighbor(core_array_size, 0, CORE_ID, -1), recv_from_pos)

                        if up_proj_reduce_line_pos > 0 and t < up_proj_reduce_line_pos:
                            A.recv(_line_neighbor(core_array_size, 0, CORE_ID, -1), CORE_ID, recv_from_neg)
                            _store_line_chunk(
                                recv_from_neg,
                                input_qkv_proj,
                                row_num=row_num,
                                chunk_length=up_all_reduce_chunk_length,
                                chunk_id=core_array_size - up_proj_reduce_line_pos + t,
                            )
                        if up_proj_reduce_line_pos < core_array_size - 1 and t < core_array_size - 1 - up_proj_reduce_line_pos:
                            A.recv(_line_neighbor(core_array_size, 0, CORE_ID, +1), CORE_ID, recv_from_pos)
                            _store_line_chunk(
                                recv_from_pos,
                                input_qkv_proj,
                                row_num=row_num,
                                chunk_length=up_all_reduce_chunk_length,
                                chunk_id=core_array_size - up_proj_reduce_line_pos - 2 - t,
                            )

    return inference
