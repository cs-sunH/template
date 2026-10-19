import math
import multiprocessing

import frontend.atlang.language as A

from frontend.hardware_parser import CloudSystemConfig
from frontend.ops.atlang.cloud import sanity_check


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


def general_spmd_autotune_inference(
    cloud_config: CloudSystemConfig,
    dtype=A.float16,
    intermediate_result_dir: str = "",
    gemm_tiling_cache_dir: str = "",
    num_workers: int = int(0.8 * multiprocessing.cpu_count()),
):
    core_num = cloud_config.chip_config.core_num
    core_array_size = int(math.sqrt(core_num))
    core_array_kwargs = {
        "system_config": cloud_config,
        "dram_row_size": 128 * 1024,
        "flit_size": cloud_config.chip_config.noc_config.flit_size,
        "inter_chip_communication_list": [],
        "general_autotune_min_tile_size": GENERAL_AUTOTUNE_MIN_TILE_SIZE,
        "general_autotune_max_candidates_per_tunable": GENERAL_AUTOTUNE_MAX_CANDIDATES_PER_TUNABLE,
        "num_workers": num_workers,
        "intermediate_result_dir": intermediate_result_dir,
        "gemm_tiling_cache_dir": gemm_tiling_cache_dir,
    }

    @A.main
    def inference(
        input_spmd: A.Tensor,
        weight0: A.Tensor,
        hidden: A.Tensor,
        weight1: A.Tensor,
        output: A.Tensor,
    ):
        with A.CoreArray(shape=(core_array_size, core_array_size), **core_array_kwargs):
            input_spmd = A.Tensor(shape=(MATRIX_M, MATRIX_KN), strides=(MATRIX_KN, 1), dtype=dtype)
            weight0 = A.Tensor(shape=(MATRIX_KN, MATRIX_KN), strides=(1, MATRIX_KN), dtype=dtype)
            hidden = A.Tensor(shape=(MATRIX_M, MATRIX_KN), strides=(MATRIX_KN, 1), dtype=dtype)
            weight1 = A.Tensor(shape=(MATRIX_KN, MATRIX_KN), strides=(1, MATRIX_KN), dtype=dtype)
            output = A.Tensor(shape=(MATRIX_M, MATRIX_KN), strides=(MATRIX_KN, 1), dtype=dtype)

            with A.SPMD(name="general_spmd_autotune", type="general"):
                with A.Kernel(
                    A.ceildiv(MATRIX_M, FIRST_TM),
                    A.ceildiv(MATRIX_KN, FIRST_TK),
                    A.ceildiv(MATRIX_KN, FIRST_TN),
                    A.ceildiv(MATRIX_M, SECOND_TM),
                    A.ceildiv(MATRIX_KN, SECOND_TK),
                    A.ceildiv(MATRIX_KN, SECOND_TN),
                    autotune=True,
                ) as (first_bM, first_bK, first_bN, second_bM, second_bK, second_bN):
                    first_tile_m = A.ceildiv(MATRIX_M, first_bM)
                    first_tile_k = A.ceildiv(MATRIX_KN, first_bK)
                    first_tile_n = A.ceildiv(MATRIX_KN, first_bN)
                    second_tile_m = A.ceildiv(MATRIX_M, second_bM)
                    second_tile_k = A.ceildiv(MATRIX_KN, second_bK)
                    second_tile_n = A.ceildiv(MATRIX_KN, second_bN)

                    first_input_tile = A.alloc(shape=(first_tile_m, first_tile_k), dtype=dtype)
                    first_weight_tile = A.alloc(shape=(first_tile_k, first_tile_n), dtype=dtype)
                    first_partial_tile = A.alloc(shape=(first_tile_m, first_tile_n), dtype=dtype)
                    first_acc_tile = A.alloc(shape=(first_tile_m, first_tile_n), dtype=dtype)
                    second_input_tile = A.alloc(shape=(second_tile_m, second_tile_k), dtype=dtype)
                    second_weight_tile = A.alloc(shape=(second_tile_k, second_tile_n), dtype=dtype)
                    second_partial_tile = A.alloc(shape=(second_tile_m, second_tile_n), dtype=dtype)
                    second_acc_tile = A.alloc(shape=(second_tile_m, second_tile_n), dtype=dtype)

                    for m in A.Serial(first_bM):
                        for n in A.Serial(first_bN):
                            A.clear(first_acc_tile)
                            for k in A.Serial(first_bK):
                                A.copy(input_spmd[m * first_tile_m, k * first_tile_k], first_input_tile)
                                A.copy(weight0[k * first_tile_k, n * first_tile_n], first_weight_tile)
                                A.gemm(first_input_tile, first_weight_tile, first_partial_tile)
                                A.add(first_acc_tile, first_partial_tile, first_acc_tile)
                            A.copy(first_acc_tile, hidden[m * first_tile_m, n * first_tile_n])

                    for m in A.Serial(second_bM):
                        for n in A.Serial(second_bN):
                            A.clear(second_acc_tile)
                            for k in A.Serial(second_bK):
                                A.copy(hidden[m * second_tile_m, k * second_tile_k], second_input_tile)
                                A.copy(weight1[k * second_tile_k, n * second_tile_n], second_weight_tile)
                                A.gemm(second_input_tile, second_weight_tile, second_partial_tile)
                                A.add(second_acc_tile, second_partial_tile, second_acc_tile)
                            A.copy(second_acc_tile, output[m * second_tile_m, n * second_tile_n])

    return inference


def general_spmd_fixed_inference(
    cloud_config: CloudSystemConfig,
    dtype=A.float16,
    intermediate_result_dir: str = "",
    gemm_tiling_cache_dir: str = "",
    num_workers: int = int(0.8 * multiprocessing.cpu_count()),
):
    core_num = cloud_config.chip_config.core_num
    core_array_size = int(math.sqrt(core_num))
    core_array_kwargs = {
        "system_config": cloud_config,
        "dram_row_size": 128 * 1024,
        "flit_size": cloud_config.chip_config.noc_config.flit_size,
        "inter_chip_communication_list": [],
        "general_autotune_min_tile_size": GENERAL_AUTOTUNE_MIN_TILE_SIZE,
        "general_autotune_max_candidates_per_tunable": GENERAL_AUTOTUNE_MAX_CANDIDATES_PER_TUNABLE,
        "num_workers": num_workers,
        "intermediate_result_dir": intermediate_result_dir,
        "gemm_tiling_cache_dir": gemm_tiling_cache_dir,
    }

    @A.main
    def inference(
        input_spmd: A.Tensor,
        weight0: A.Tensor,
        hidden: A.Tensor,
        weight1: A.Tensor,
        output: A.Tensor,
    ):
        with A.CoreArray(shape=(core_array_size, core_array_size), **core_array_kwargs):
            input_spmd = A.Tensor(shape=(MATRIX_M, MATRIX_KN), strides=(MATRIX_KN, 1), dtype=dtype)
            weight0 = A.Tensor(shape=(MATRIX_KN, MATRIX_KN), strides=(1, MATRIX_KN), dtype=dtype)
            hidden = A.Tensor(shape=(MATRIX_M, MATRIX_KN), strides=(MATRIX_KN, 1), dtype=dtype)
            weight1 = A.Tensor(shape=(MATRIX_KN, MATRIX_KN), strides=(1, MATRIX_KN), dtype=dtype)
            output = A.Tensor(shape=(MATRIX_M, MATRIX_KN), strides=(MATRIX_KN, 1), dtype=dtype)

            with A.SPMD(name="general_spmd_fixed", type="general"):
                with A.Kernel(
                    A.ceildiv(MATRIX_M, FIRST_TM),
                    A.ceildiv(MATRIX_KN, FIRST_TK),
                    A.ceildiv(MATRIX_KN, FIRST_TN),
                    A.ceildiv(MATRIX_M, SECOND_TM),
                    A.ceildiv(MATRIX_KN, SECOND_TK),
                    A.ceildiv(MATRIX_KN, SECOND_TN),
                    autotune=False,
                ) as (first_bM, first_bK, first_bN, second_bM, second_bK, second_bN):
                    first_tile_m = A.ceildiv(MATRIX_M, first_bM)
                    first_tile_k = A.ceildiv(MATRIX_KN, first_bK)
                    first_tile_n = A.ceildiv(MATRIX_KN, first_bN)
                    second_tile_m = A.ceildiv(MATRIX_M, second_bM)
                    second_tile_k = A.ceildiv(MATRIX_KN, second_bK)
                    second_tile_n = A.ceildiv(MATRIX_KN, second_bN)

                    first_input_tile = A.alloc(shape=(first_tile_m, first_tile_k), dtype=dtype)
                    first_weight_tile = A.alloc(shape=(first_tile_k, first_tile_n), dtype=dtype)
                    first_partial_tile = A.alloc(shape=(first_tile_m, first_tile_n), dtype=dtype)
                    first_acc_tile = A.alloc(shape=(first_tile_m, first_tile_n), dtype=dtype)
                    second_input_tile = A.alloc(shape=(second_tile_m, second_tile_k), dtype=dtype)
                    second_weight_tile = A.alloc(shape=(second_tile_k, second_tile_n), dtype=dtype)
                    second_partial_tile = A.alloc(shape=(second_tile_m, second_tile_n), dtype=dtype)
                    second_acc_tile = A.alloc(shape=(second_tile_m, second_tile_n), dtype=dtype)

                    for m in A.Serial(first_bM):
                        for n in A.Serial(first_bN):
                            A.clear(first_acc_tile)
                            for k in A.Serial(first_bK):
                                A.copy(input_spmd[m * first_tile_m, k * first_tile_k], first_input_tile)
                                A.copy(weight0[k * first_tile_k, n * first_tile_n], first_weight_tile)
                                A.gemm(first_input_tile, first_weight_tile, first_partial_tile)
                                A.add(first_acc_tile, first_partial_tile, first_acc_tile)
                            A.copy(first_acc_tile, hidden[m * first_tile_m, n * first_tile_n])

                    for m in A.Serial(second_bM):
                        for n in A.Serial(second_bN):
                            A.clear(second_acc_tile)
                            for k in A.Serial(second_bK):
                                A.copy(hidden[m * second_tile_m, k * second_tile_k], second_input_tile)
                                A.copy(weight1[k * second_tile_k, n * second_tile_n], second_weight_tile)
                                A.gemm(second_input_tile, second_weight_tile, second_partial_tile)
                                A.add(second_acc_tile, second_partial_tile, second_acc_tile)
                            A.copy(second_acc_tile, output[m * second_tile_m, n * second_tile_n])

    return inference


__all__ = ["general_spmd_autotune_inference", "general_spmd_fixed_inference", "sanity_check"]
