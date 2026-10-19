import builtins
import math
import multiprocessing

import tilelang
import tilelang.language as T

from frontend.hardware_parser import CloudSystemConfig
from frontend.ops.tilelang.cloud import sanity_check


_python_range = builtins.range

MIN_TM = 4
MIN_TK = 32
MIN_TN = 32
MATRIX_M = 16
MATRIX_KN = 512
GENERAL_AUTOTUNE_MIN_TILE_SIZE = 16
GENERAL_AUTOTUNE_MAX_CANDIDATES_PER_TUNABLE = 5


@tilelang.jit(simulator=True)
def general_mpmd_autotune_inference(
    cloud_config: CloudSystemConfig,
    dtype=T.float16,
    intermediate_result_dir: str = "",
    gemm_tiling_cache_dir: str = "",
    num_workers: int = int(0.8 * multiprocessing.cpu_count()),
):
    element_size = 2 if dtype == T.float16 else -1

    core_num = cloud_config.chip_config.core_num
    core_array_size = int(math.sqrt(core_num))
    left_column_cores = [
        row * core_array_size + col
        for row in _python_range(core_array_size)
        for col in (0, 1)
    ]
    right_column_cores = [
        row * core_array_size + col
        for row in _python_range(core_array_size)
        for col in (2, 3)
    ]
    core_array_kwargs = {
        "system_config": cloud_config,
        "dram_row_size": 128 * 1024,
        "flit_size": cloud_config.chip_config.noc_config.flit_size,
        "element_size": element_size,
        "inter_chip_communication_list": [],
        "general_autotune_min_tile_size": GENERAL_AUTOTUNE_MIN_TILE_SIZE,
        "general_autotune_max_candidates_per_tunable": GENERAL_AUTOTUNE_MAX_CANDIDATES_PER_TUNABLE,
        "num_workers": num_workers,
        "intermediate_result_dir": intermediate_result_dir,
        "gemm_tiling_cache_dir": gemm_tiling_cache_dir,
    }

    @T.prim_func
    def inference(
        mpmd_input: T.Tensor,
        mpmd_weight0: T.Tensor,
        mpmd_partial: T.Tensor,
        mpmd_weight1: T.Tensor,
        mpmd_output: T.Tensor,
    ):
        with T.CoreArray(shape=(core_array_size, core_array_size), **core_array_kwargs):
            mpmd_input = T.Tensor(shape=(MATRIX_M, MATRIX_KN), strides=(MATRIX_KN, 1), dtype=dtype)
            mpmd_weight0 = T.Tensor(shape=(MATRIX_KN, MATRIX_KN), strides=(1, MATRIX_KN), dtype=dtype)
            mpmd_partial = T.Tensor(shape=(MATRIX_M, MATRIX_KN), strides=(MATRIX_KN, 1), dtype=dtype)
            mpmd_weight1 = T.Tensor(shape=(MATRIX_KN, MATRIX_KN), strides=(1, MATRIX_KN), dtype=dtype)
            mpmd_output = T.Tensor(shape=(MATRIX_M, MATRIX_KN), strides=(MATRIX_KN, 1), dtype=dtype)

            with T.MPMD(name="general_mpmd_autotune_gemm", type="general") as CORE_ID:
                with T.Kernel(
                    T.ceildiv(MATRIX_M, MIN_TM),
                    T.ceildiv(MATRIX_KN, MIN_TK),
                    T.ceildiv(MATRIX_KN, MIN_TN),
                    autotune=True,
                    core_list=left_column_cores,
                ) as (bM, bK, bN):
                    tile_m = T.ceildiv(MATRIX_M, bM)
                    tile_k = T.ceildiv(MATRIX_KN, bK)
                    tile_n = T.ceildiv(MATRIX_KN, bN)
                    input_tile = T.alloc(shape=(tile_m, tile_k), dtype=dtype)
                    weight_tile = T.alloc(shape=(tile_k, tile_n), dtype=dtype)
                    partial_tile = T.alloc(shape=(tile_m, tile_n), dtype=dtype)
                    output_tile = T.alloc(shape=(tile_m, tile_n), dtype=dtype)

                    for m in T.Serial(bM):
                        for n in T.Serial(bN):
                            T.clear(output_tile)
                            for k in T.Serial(bK):
                                T.copy(mpmd_input[m * tile_m, k * tile_k], input_tile)
                                T.copy(mpmd_weight0[k * tile_k, n * tile_n], weight_tile)
                                T.gemm(input_tile, weight_tile, partial_tile)
                                T.add(output_tile, partial_tile, output_tile)
                            T.copy(output_tile, mpmd_partial[m * tile_m, n * tile_n])

            with T.MPMD(name="general_mpmd_autotune_reduce_gemm", type="general") as CORE_ID:
                with T.Kernel(
                    T.ceildiv(MATRIX_M, MIN_TM),
                    T.ceildiv(MATRIX_KN, MIN_TK),
                    T.ceildiv(MATRIX_KN, MIN_TN),
                    autotune=True,
                    core_list=left_column_cores,
                ) as (bM, bK, bN):
                    tile_m = T.ceildiv(MATRIX_M, bM)
                    tile_k = T.ceildiv(MATRIX_KN, bK)
                    tile_n = T.ceildiv(MATRIX_KN, bN)
                    reduce_tile = T.alloc(shape=(tile_m, tile_k), dtype=dtype)
                    up_reduce_tile = T.alloc(shape=(tile_m, tile_k), dtype=dtype)
                    down_reduce_tile = T.alloc(shape=(tile_m, tile_k), dtype=dtype)
                    recv_from_down_tile = T.alloc(shape=(tile_m, tile_k), dtype=dtype)
                    recv_from_up_tile = T.alloc(shape=(tile_m, tile_k), dtype=dtype)
                    input_tile = T.alloc(shape=(tile_m, tile_k), dtype=dtype)
                    weight_tile = T.alloc(shape=(tile_k, tile_n), dtype=dtype)
                    partial_tile = T.alloc(shape=(tile_m, tile_n), dtype=dtype)
                    output_tile = T.alloc(shape=(tile_m, tile_n), dtype=dtype)
                    core_row = CORE_ID // core_array_size

                    for m in T.Serial(bM):
                        for k in T.Serial(bK):
                            T.copy(mpmd_partial[m * tile_m, k * tile_k], reduce_tile)
                            T.copy(reduce_tile, up_reduce_tile)
                            T.copy(reduce_tile, down_reduce_tile)
                            for step in T.Serial(core_array_size - 1):
                                if core_row > 0 and core_row <= core_array_size - 1 - step:
                                    T.send(CORE_ID, CORE_ID - core_array_size, up_reduce_tile)
                                if core_row >= step and core_row < core_array_size - 1:
                                    T.send(CORE_ID, CORE_ID + core_array_size, down_reduce_tile)
                                if core_row < core_array_size - 1 - step:
                                    T.recv(CORE_ID + core_array_size, CORE_ID, recv_from_down_tile)
                                    T.add(recv_from_down_tile, reduce_tile, reduce_tile)
                                    T.copy(recv_from_down_tile, up_reduce_tile)
                                if core_row > step:
                                    T.recv(CORE_ID - core_array_size, CORE_ID, recv_from_up_tile)
                                    T.add(recv_from_up_tile, reduce_tile, reduce_tile)
                                    T.copy(recv_from_up_tile, down_reduce_tile)
                            T.copy(reduce_tile, mpmd_partial[m * tile_m, k * tile_k])

                    for m in T.Serial(bM):
                        for n in T.Serial(bN):
                            T.clear(output_tile)
                            for k in T.Serial(bK):
                                T.copy(mpmd_partial[m * tile_m, k * tile_k], input_tile)
                                T.copy(mpmd_weight1[k * tile_k, n * tile_n], weight_tile)
                                T.gemm(input_tile, weight_tile, partial_tile)
                                T.add(output_tile, partial_tile, output_tile)
                            T.copy(output_tile, mpmd_output[m * tile_m, n * tile_n])

                with T.Kernel(
                    T.ceildiv(MATRIX_M, MIN_TM),
                    T.ceildiv(MATRIX_KN, MIN_TK),
                    T.ceildiv(MATRIX_KN, MIN_TN),
                    autotune=True,
                    core_list=right_column_cores,
                ) as (bM, bK, bN):
                    tile_m = T.ceildiv(MATRIX_M, bM)
                    tile_k = T.ceildiv(MATRIX_KN, bK)
                    tile_n = T.ceildiv(MATRIX_KN, bN)
                    input_tile = T.alloc(shape=(tile_m, tile_k), dtype=dtype)
                    weight_tile = T.alloc(shape=(tile_k, tile_n), dtype=dtype)
                    partial_tile = T.alloc(shape=(tile_m, tile_n), dtype=dtype)
                    output_tile = T.alloc(shape=(tile_m, tile_n), dtype=dtype)
                    up_output_tile = T.alloc(shape=(tile_m, tile_n), dtype=dtype)
                    down_output_tile = T.alloc(shape=(tile_m, tile_n), dtype=dtype)
                    recv_from_down_tile = T.alloc(shape=(tile_m, tile_n), dtype=dtype)
                    recv_from_up_tile = T.alloc(shape=(tile_m, tile_n), dtype=dtype)
                    core_row = CORE_ID // core_array_size

                    for m in T.Serial(bM):
                        for n in T.Serial(bN):
                            T.clear(output_tile)
                            for k in T.Serial(bK):
                                T.copy(mpmd_input[m * tile_m, k * tile_k], input_tile)
                                T.copy(mpmd_weight0[k * tile_k, n * tile_n], weight_tile)
                                T.gemm(input_tile, weight_tile, partial_tile)
                                T.add(output_tile, partial_tile, output_tile)
                            T.copy(output_tile, mpmd_output[m * tile_m, n * tile_n])

                    for m in T.Serial(bM):
                        for n in T.Serial(bN):
                            T.copy(mpmd_output[m * tile_m, n * tile_n], output_tile)
                            T.copy(output_tile, up_output_tile)
                            T.copy(output_tile, down_output_tile)
                            for step in T.Serial(core_array_size - 1):
                                if core_row > 0 and core_row <= core_array_size - 1 - step:
                                    T.send(CORE_ID, CORE_ID - core_array_size, up_output_tile)
                                if core_row >= step and core_row < core_array_size - 1:
                                    T.send(CORE_ID, CORE_ID + core_array_size, down_output_tile)
                                if core_row < core_array_size - 1 - step:
                                    T.recv(CORE_ID + core_array_size, CORE_ID, recv_from_down_tile)
                                    T.add(recv_from_down_tile, output_tile, output_tile)
                                    T.copy(recv_from_down_tile, up_output_tile)
                                if core_row > step:
                                    T.recv(CORE_ID - core_array_size, CORE_ID, recv_from_up_tile)
                                    T.add(recv_from_up_tile, output_tile, output_tile)
                                    T.copy(recv_from_up_tile, down_output_tile)
                            T.copy(output_tile, mpmd_output[m * tile_m, n * tile_n])

    return inference


@tilelang.jit(simulator=True)
def general_mpmd_fixed_inference(
    cloud_config: CloudSystemConfig,
    dtype=T.float16,
    intermediate_result_dir: str = "",
    gemm_tiling_cache_dir: str = "",
    num_workers: int = int(0.8 * multiprocessing.cpu_count()),
):
    element_size = 2 if dtype == T.float16 else -1

    core_num = cloud_config.chip_config.core_num
    core_array_size = int(math.sqrt(core_num))
    left_column_cores = [
        row * core_array_size + col
        for row in _python_range(core_array_size)
        for col in (0, 1)
    ]
    right_column_cores = [
        row * core_array_size + col
        for row in _python_range(core_array_size)
        for col in (2, 3)
    ]
    core_array_kwargs = {
        "system_config": cloud_config,
        "dram_row_size": 128 * 1024,
        "flit_size": cloud_config.chip_config.noc_config.flit_size,
        "element_size": element_size,
        "inter_chip_communication_list": [],
        "general_autotune_min_tile_size": GENERAL_AUTOTUNE_MIN_TILE_SIZE,
        "general_autotune_max_candidates_per_tunable": GENERAL_AUTOTUNE_MAX_CANDIDATES_PER_TUNABLE,
        "num_workers": num_workers,
        "intermediate_result_dir": intermediate_result_dir,
        "gemm_tiling_cache_dir": gemm_tiling_cache_dir,
    }

    @T.prim_func
    def inference(
        mpmd_input: T.Tensor,
        mpmd_weight0: T.Tensor,
        mpmd_partial: T.Tensor,
        mpmd_weight1: T.Tensor,
        mpmd_output: T.Tensor,
    ):
        with T.CoreArray(shape=(core_array_size, core_array_size), **core_array_kwargs):
            mpmd_input = T.Tensor(shape=(MATRIX_M, MATRIX_KN), strides=(MATRIX_KN, 1), dtype=dtype)
            mpmd_weight0 = T.Tensor(shape=(MATRIX_KN, MATRIX_KN), strides=(1, MATRIX_KN), dtype=dtype)
            mpmd_partial = T.Tensor(shape=(MATRIX_M, MATRIX_KN), strides=(MATRIX_KN, 1), dtype=dtype)
            mpmd_weight1 = T.Tensor(shape=(MATRIX_KN, MATRIX_KN), strides=(1, MATRIX_KN), dtype=dtype)
            mpmd_output = T.Tensor(shape=(MATRIX_M, MATRIX_KN), strides=(MATRIX_KN, 1), dtype=dtype)

            with T.MPMD(name="general_mpmd_fixed_gemm", type="general") as CORE_ID:
                with T.Kernel(
                    T.ceildiv(MATRIX_M, MIN_TM),
                    T.ceildiv(MATRIX_KN, MIN_TK),
                    T.ceildiv(MATRIX_KN, MIN_TN),
                    autotune=False,
                    core_list=left_column_cores,
                ) as (bM, bK, bN):
                    tile_m = T.ceildiv(MATRIX_M, bM)
                    tile_k = T.ceildiv(MATRIX_KN, bK)
                    tile_n = T.ceildiv(MATRIX_KN, bN)
                    input_tile = T.alloc(shape=(tile_m, tile_k), dtype=dtype)
                    weight_tile = T.alloc(shape=(tile_k, tile_n), dtype=dtype)
                    partial_tile = T.alloc(shape=(tile_m, tile_n), dtype=dtype)
                    output_tile = T.alloc(shape=(tile_m, tile_n), dtype=dtype)

                    for m in T.Serial(bM):
                        for n in T.Serial(bN):
                            T.clear(output_tile)
                            for k in T.Serial(bK):
                                T.copy(mpmd_input[m * tile_m, k * tile_k], input_tile)
                                T.copy(mpmd_weight0[k * tile_k, n * tile_n], weight_tile)
                                T.gemm(input_tile, weight_tile, partial_tile)
                                T.add(output_tile, partial_tile, output_tile)
                            T.copy(output_tile, mpmd_partial[m * tile_m, n * tile_n])

            with T.MPMD(name="general_mpmd_fixed_reduce_gemm", type="general") as CORE_ID:
                with T.Kernel(
                    T.ceildiv(MATRIX_M, MIN_TM),
                    T.ceildiv(MATRIX_KN, MIN_TK),
                    T.ceildiv(MATRIX_KN, MIN_TN),
                    autotune=False,
                    core_list=left_column_cores,
                ) as (bM, bK, bN):
                    tile_m = T.ceildiv(MATRIX_M, bM)
                    tile_k = T.ceildiv(MATRIX_KN, bK)
                    tile_n = T.ceildiv(MATRIX_KN, bN)
                    reduce_tile = T.alloc(shape=(tile_m, tile_k), dtype=dtype)
                    up_reduce_tile = T.alloc(shape=(tile_m, tile_k), dtype=dtype)
                    down_reduce_tile = T.alloc(shape=(tile_m, tile_k), dtype=dtype)
                    recv_from_down_tile = T.alloc(shape=(tile_m, tile_k), dtype=dtype)
                    recv_from_up_tile = T.alloc(shape=(tile_m, tile_k), dtype=dtype)
                    input_tile = T.alloc(shape=(tile_m, tile_k), dtype=dtype)
                    weight_tile = T.alloc(shape=(tile_k, tile_n), dtype=dtype)
                    partial_tile = T.alloc(shape=(tile_m, tile_n), dtype=dtype)
                    output_tile = T.alloc(shape=(tile_m, tile_n), dtype=dtype)
                    core_row = CORE_ID // core_array_size

                    for m in T.Serial(bM):
                        for k in T.Serial(bK):
                            T.copy(mpmd_partial[m * tile_m, k * tile_k], reduce_tile)
                            T.copy(reduce_tile, up_reduce_tile)
                            T.copy(reduce_tile, down_reduce_tile)
                            for step in T.Serial(core_array_size - 1):
                                if core_row > 0 and core_row <= core_array_size - 1 - step:
                                    T.send(CORE_ID, CORE_ID - core_array_size, up_reduce_tile)
                                if core_row >= step and core_row < core_array_size - 1:
                                    T.send(CORE_ID, CORE_ID + core_array_size, down_reduce_tile)
                                if core_row < core_array_size - 1 - step:
                                    T.recv(CORE_ID + core_array_size, CORE_ID, recv_from_down_tile)
                                    T.add(recv_from_down_tile, reduce_tile, reduce_tile)
                                    T.copy(recv_from_down_tile, up_reduce_tile)
                                if core_row > step:
                                    T.recv(CORE_ID - core_array_size, CORE_ID, recv_from_up_tile)
                                    T.add(recv_from_up_tile, reduce_tile, reduce_tile)
                                    T.copy(recv_from_up_tile, down_reduce_tile)
                            T.copy(reduce_tile, mpmd_partial[m * tile_m, k * tile_k])

                    for m in T.Serial(bM):
                        for n in T.Serial(bN):
                            T.clear(output_tile)
                            for k in T.Serial(bK):
                                T.copy(mpmd_partial[m * tile_m, k * tile_k], input_tile)
                                T.copy(mpmd_weight1[k * tile_k, n * tile_n], weight_tile)
                                T.gemm(input_tile, weight_tile, partial_tile)
                                T.add(output_tile, partial_tile, output_tile)
                            T.copy(output_tile, mpmd_output[m * tile_m, n * tile_n])

                with T.Kernel(
                    T.ceildiv(MATRIX_M, MIN_TM),
                    T.ceildiv(MATRIX_KN, MIN_TK),
                    T.ceildiv(MATRIX_KN, MIN_TN),
                    autotune=False,
                    core_list=right_column_cores,
                ) as (bM, bK, bN):
                    tile_m = T.ceildiv(MATRIX_M, bM)
                    tile_k = T.ceildiv(MATRIX_KN, bK)
                    tile_n = T.ceildiv(MATRIX_KN, bN)
                    input_tile = T.alloc(shape=(tile_m, tile_k), dtype=dtype)
                    weight_tile = T.alloc(shape=(tile_k, tile_n), dtype=dtype)
                    partial_tile = T.alloc(shape=(tile_m, tile_n), dtype=dtype)
                    output_tile = T.alloc(shape=(tile_m, tile_n), dtype=dtype)
                    up_output_tile = T.alloc(shape=(tile_m, tile_n), dtype=dtype)
                    down_output_tile = T.alloc(shape=(tile_m, tile_n), dtype=dtype)
                    recv_from_down_tile = T.alloc(shape=(tile_m, tile_n), dtype=dtype)
                    recv_from_up_tile = T.alloc(shape=(tile_m, tile_n), dtype=dtype)
                    core_row = CORE_ID // core_array_size

                    for m in T.Serial(bM):
                        for n in T.Serial(bN):
                            T.clear(output_tile)
                            for k in T.Serial(bK):
                                T.copy(mpmd_input[m * tile_m, k * tile_k], input_tile)
                                T.copy(mpmd_weight0[k * tile_k, n * tile_n], weight_tile)
                                T.gemm(input_tile, weight_tile, partial_tile)
                                T.add(output_tile, partial_tile, output_tile)
                            T.copy(output_tile, mpmd_output[m * tile_m, n * tile_n])

                    for m in T.Serial(bM):
                        for n in T.Serial(bN):
                            T.copy(mpmd_output[m * tile_m, n * tile_n], output_tile)
                            T.copy(output_tile, up_output_tile)
                            T.copy(output_tile, down_output_tile)
                            for step in T.Serial(core_array_size - 1):
                                if core_row > 0 and core_row <= core_array_size - 1 - step:
                                    T.send(CORE_ID, CORE_ID - core_array_size, up_output_tile)
                                if core_row >= step and core_row < core_array_size - 1:
                                    T.send(CORE_ID, CORE_ID + core_array_size, down_output_tile)
                                if core_row < core_array_size - 1 - step:
                                    T.recv(CORE_ID + core_array_size, CORE_ID, recv_from_down_tile)
                                    T.add(recv_from_down_tile, output_tile, output_tile)
                                    T.copy(recv_from_down_tile, up_output_tile)
                                if core_row > step:
                                    T.recv(CORE_ID - core_array_size, CORE_ID, recv_from_up_tile)
                                    T.add(recv_from_up_tile, output_tile, output_tile)
                                    T.copy(recv_from_up_tile, down_output_tile)
                            T.copy(output_tile, mpmd_output[m * tile_m, n * tile_n])

    return inference


__all__ = ["general_mpmd_autotune_inference", "general_mpmd_fixed_inference", "sanity_check"]
