import math
import multiprocessing

import tilelang
import tilelang.language as T

from frontend.hardware_parser import CloudSystemConfig
from frontend.ops.tilelang.cloud import sanity_check


FIRST_TM = 4
FIRST_TK = 16
FIRST_TN = 16
SECOND_TM = 4
SECOND_TK = 32
SECOND_TN = 32
MATRIX_M = 16
MATRIX_KN = 1024
GENERAL_AUTOTUNE_MIN_TILE_SIZE = 16
GENERAL_AUTOTUNE_MAX_CANDIDATES_PER_TUNABLE = 5


@tilelang.jit(simulator=True)
def general_spmd_autotune_inference(
    cloud_config: CloudSystemConfig,
    dtype=T.float16,
    intermediate_result_dir: str = "",
    gemm_tiling_cache_dir: str = "",
    num_workers: int = int(0.8 * multiprocessing.cpu_count()),
):
    element_size = 2 if dtype == T.float16 else -1

    core_num = cloud_config.chip_config.core_num
    core_array_size = int(math.sqrt(core_num))
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
        input_spmd: T.Tensor,
        weight0: T.Tensor,
        hidden: T.Tensor,
        weight1: T.Tensor,
        output: T.Tensor,
    ):
        with T.CoreArray(shape=(core_array_size, core_array_size), **core_array_kwargs):
            input_spmd = T.Tensor(shape=(MATRIX_M, MATRIX_KN), strides=(MATRIX_KN, 1), dtype=dtype)
            weight0 = T.Tensor(shape=(MATRIX_KN, MATRIX_KN), strides=(1, MATRIX_KN), dtype=dtype)
            hidden = T.Tensor(shape=(MATRIX_M, MATRIX_KN), strides=(MATRIX_KN, 1), dtype=dtype)
            weight1 = T.Tensor(shape=(MATRIX_KN, MATRIX_KN), strides=(1, MATRIX_KN), dtype=dtype)
            output = T.Tensor(shape=(MATRIX_M, MATRIX_KN), strides=(MATRIX_KN, 1), dtype=dtype)

            with T.SPMD(name="general_spmd_autotune", type="general"):
                with T.Kernel(
                    T.ceildiv(MATRIX_M, FIRST_TM),
                    T.ceildiv(MATRIX_KN, FIRST_TK),
                    T.ceildiv(MATRIX_KN, FIRST_TN),
                    T.ceildiv(MATRIX_M, SECOND_TM),
                    T.ceildiv(MATRIX_KN, SECOND_TK),
                    T.ceildiv(MATRIX_KN, SECOND_TN),
                    autotune=True,
                ) as (first_bM, first_bK, first_bN, second_bM, second_bK, second_bN):
                    first_tile_m = T.ceildiv(MATRIX_M, first_bM)
                    first_tile_k = T.ceildiv(MATRIX_KN, first_bK)
                    first_tile_n = T.ceildiv(MATRIX_KN, first_bN)
                    second_tile_m = T.ceildiv(MATRIX_M, second_bM)
                    second_tile_k = T.ceildiv(MATRIX_KN, second_bK)
                    second_tile_n = T.ceildiv(MATRIX_KN, second_bN)

                    first_input_tile = T.alloc(shape=(first_tile_m, first_tile_k), dtype=dtype)
                    first_weight_tile = T.alloc(shape=(first_tile_k, first_tile_n), dtype=dtype)
                    first_partial_tile = T.alloc(shape=(first_tile_m, first_tile_n), dtype=dtype)
                    first_acc_tile = T.alloc(shape=(first_tile_m, first_tile_n), dtype=dtype)
                    second_input_tile = T.alloc(shape=(second_tile_m, second_tile_k), dtype=dtype)
                    second_weight_tile = T.alloc(shape=(second_tile_k, second_tile_n), dtype=dtype)
                    second_partial_tile = T.alloc(shape=(second_tile_m, second_tile_n), dtype=dtype)
                    second_acc_tile = T.alloc(shape=(second_tile_m, second_tile_n), dtype=dtype)

                    for m in T.Serial(first_bM):
                        for n in T.Serial(first_bN):
                            T.clear(first_acc_tile)
                            for k in T.Serial(first_bK):
                                T.copy(input_spmd[m * first_tile_m, k * first_tile_k], first_input_tile)
                                T.copy(weight0[k * first_tile_k, n * first_tile_n], first_weight_tile)
                                T.gemm(first_input_tile, first_weight_tile, first_partial_tile)
                                T.add(first_acc_tile, first_partial_tile, first_acc_tile)
                            T.copy(first_acc_tile, hidden[m * first_tile_m, n * first_tile_n])

                    for m in T.Serial(second_bM):
                        for n in T.Serial(second_bN):
                            T.clear(second_acc_tile)
                            for k in T.Serial(second_bK):
                                T.copy(hidden[m * second_tile_m, k * second_tile_k], second_input_tile)
                                T.copy(weight1[k * second_tile_k, n * second_tile_n], second_weight_tile)
                                T.gemm(second_input_tile, second_weight_tile, second_partial_tile)
                                T.add(second_acc_tile, second_partial_tile, second_acc_tile)
                            T.copy(second_acc_tile, output[m * second_tile_m, n * second_tile_n])

    return inference


@tilelang.jit(simulator=True)
def general_spmd_fixed_inference(
    cloud_config: CloudSystemConfig,
    dtype=T.float16,
    intermediate_result_dir: str = "",
    gemm_tiling_cache_dir: str = "",
    num_workers: int = int(0.8 * multiprocessing.cpu_count()),
):
    element_size = 2 if dtype == T.float16 else -1

    core_num = cloud_config.chip_config.core_num
    core_array_size = int(math.sqrt(core_num))
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
        input_spmd: T.Tensor,
        weight0: T.Tensor,
        hidden: T.Tensor,
        weight1: T.Tensor,
        output: T.Tensor,
    ):
        with T.CoreArray(shape=(core_array_size, core_array_size), **core_array_kwargs):
            input_spmd = T.Tensor(shape=(MATRIX_M, MATRIX_KN), strides=(MATRIX_KN, 1), dtype=dtype)
            weight0 = T.Tensor(shape=(MATRIX_KN, MATRIX_KN), strides=(1, MATRIX_KN), dtype=dtype)
            hidden = T.Tensor(shape=(MATRIX_M, MATRIX_KN), strides=(MATRIX_KN, 1), dtype=dtype)
            weight1 = T.Tensor(shape=(MATRIX_KN, MATRIX_KN), strides=(1, MATRIX_KN), dtype=dtype)
            output = T.Tensor(shape=(MATRIX_M, MATRIX_KN), strides=(MATRIX_KN, 1), dtype=dtype)

            with T.SPMD(name="general_spmd_fixed", type="general"):
                with T.Kernel(
                    T.ceildiv(MATRIX_M, FIRST_TM),
                    T.ceildiv(MATRIX_KN, FIRST_TK),
                    T.ceildiv(MATRIX_KN, FIRST_TN),
                    T.ceildiv(MATRIX_M, SECOND_TM),
                    T.ceildiv(MATRIX_KN, SECOND_TK),
                    T.ceildiv(MATRIX_KN, SECOND_TN),
                    autotune=False,
                ) as (first_bM, first_bK, first_bN, second_bM, second_bK, second_bN):
                    first_tile_m = T.ceildiv(MATRIX_M, first_bM)
                    first_tile_k = T.ceildiv(MATRIX_KN, first_bK)
                    first_tile_n = T.ceildiv(MATRIX_KN, first_bN)
                    second_tile_m = T.ceildiv(MATRIX_M, second_bM)
                    second_tile_k = T.ceildiv(MATRIX_KN, second_bK)
                    second_tile_n = T.ceildiv(MATRIX_KN, second_bN)

                    first_input_tile = T.alloc(shape=(first_tile_m, first_tile_k), dtype=dtype)
                    first_weight_tile = T.alloc(shape=(first_tile_k, first_tile_n), dtype=dtype)
                    first_partial_tile = T.alloc(shape=(first_tile_m, first_tile_n), dtype=dtype)
                    first_acc_tile = T.alloc(shape=(first_tile_m, first_tile_n), dtype=dtype)
                    second_input_tile = T.alloc(shape=(second_tile_m, second_tile_k), dtype=dtype)
                    second_weight_tile = T.alloc(shape=(second_tile_k, second_tile_n), dtype=dtype)
                    second_partial_tile = T.alloc(shape=(second_tile_m, second_tile_n), dtype=dtype)
                    second_acc_tile = T.alloc(shape=(second_tile_m, second_tile_n), dtype=dtype)

                    for m in T.Serial(first_bM):
                        for n in T.Serial(first_bN):
                            T.clear(first_acc_tile)
                            for k in T.Serial(first_bK):
                                T.copy(input_spmd[m * first_tile_m, k * first_tile_k], first_input_tile)
                                T.copy(weight0[k * first_tile_k, n * first_tile_n], first_weight_tile)
                                T.gemm(first_input_tile, first_weight_tile, first_partial_tile)
                                T.add(first_acc_tile, first_partial_tile, first_acc_tile)
                            T.copy(first_acc_tile, hidden[m * first_tile_m, n * first_tile_n])

                    for m in T.Serial(second_bM):
                        for n in T.Serial(second_bN):
                            T.clear(second_acc_tile)
                            for k in T.Serial(second_bK):
                                T.copy(hidden[m * second_tile_m, k * second_tile_k], second_input_tile)
                                T.copy(weight1[k * second_tile_k, n * second_tile_n], second_weight_tile)
                                T.gemm(second_input_tile, second_weight_tile, second_partial_tile)
                                T.add(second_acc_tile, second_partial_tile, second_acc_tile)
                            T.copy(second_acc_tile, output[m * second_tile_m, n * second_tile_n])

    return inference


__all__ = ["general_spmd_autotune_inference", "general_spmd_fixed_inference", "sanity_check"]
