import os
import argparse
import multiprocessing
import heapq
import random
import itertools
import yaml
import math
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from frontend.model_parser import *
from frontend.util import set_pdeathsig


ATTENTION_ROUND_COUNT = 10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-type",
        type=str,
        default="model_moe",
        choices=["attention", "matrix", "model_dense", "model_moe"]
    )
    parser.add_argument("--config-path", type=str, default="configs/architecture/chip/test_chip_16ch.yaml")
    parser.add_argument("--mid-dir", type=str, default="decaparated/dram_test/mid_results")
    parser.add_argument("--output-dir", type=str, default="decaparated/dram_test/output_results")
    parser.add_argument("--num-workers", type=int, default=int(multiprocessing.cpu_count()*0.8))
    return parser.parse_args()


def partition_list(numbers, n):
    sorted_numbers = sorted(numbers, reverse=True)
    heap = [(0, i, []) for i in range(n)]
    heapq.heapify(heap)

    for num in sorted_numbers:
        current_sum, idx, sublist = heapq.heappop(heap)
        sublist.append(num)
        if type(num) in [int, float]:
            current_sum += num
        elif type(num) in [list, tuple]:
            products = 1
            for x in num:
                assert type(x) in [int, float]
                products *= x
            current_sum += products
        else:
            raise ValueError(f"Invalid number type: {type(num)}")
        heapq.heappush(heap, (current_sum, idx, sublist))

    result = [item[2] for item in sorted(heap, key=lambda x: x[1])]
    return result


def partition_by_count(items, n):
    partitions = [[] for _ in range(n)]
    for idx, item in enumerate(items):
        partitions[idx % n].append(item)
    return partitions


def get_factors(n, min_factor=1):
    factors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            if i >= min_factor:
                factors.append(i)
            if i != n // i and n // i >= min_factor:
                factors.append(n // i)
    return sorted(factors)


def set_shape(
    args,
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    attention_block: List[Operator],
    ffn_moe_block: List[Operator],
):
    chip_config = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
    core_num = chip_config["architecture"]["core_num"]
    core_array_size = int(math.sqrt(chip_config["architecture"]["core_num"]))
    assert core_array_size**2 == chip_config["architecture"]["core_num"],\
        f"Currently we assume core array is square, but got {core_array_size**2} != {core_num}"

    prompt_length = []
    context_length = [args.context_length for _ in range(args.batch_size)]
    total_token_num = sum(prompt_length) + len(context_length)
    for operator in attention_block:
        if operator.op_type == OperatorType.GEMM:
            operator.M = total_token_num
            operator.N = math.ceil(operator.N/core_array_size)
            operator.K = math.ceil(operator.K/core_array_size)
        elif operator.op_type == OperatorType.ATTENTION:
            operator.input_length = [math.ceil(p/core_num) for p in prompt_length] + [1 for _ in range(len(context_length))]
            operator.context_length = [math.ceil(p/core_num) for p in prompt_length] + [math.ceil(c/core_num) for c in context_length]
    
        if operator.name == "uk_proj":
            operator.K *= operator.B
            operator.B = 1
        if operator.name == "uv_proj":
            operator.N *= operator.B
            operator.B = 1

    if model_config.is_moe:
        per_ffn_token_num = math.ceil(total_token_num * (model_config.moe_top_k + model_config.moe_shared_experts) / model_config.moe_num_experts)
        per_device_ffn_num = math.ceil(min(total_token_num * (model_config.moe_top_k + model_config.moe_shared_experts), model_config.moe_num_experts) / parallel_config.ep_size)
    else:
        per_device_ffn_num = 1
        per_ffn_token_num = total_token_num
    for operator in ffn_moe_block:
        if operator.op_type == OperatorType.GEMM:
            operator.B = per_device_ffn_num
            operator.M = per_ffn_token_num
            operator.N = math.ceil(operator.N/core_array_size)
            operator.K = math.ceil(operator.K/core_array_size)


def attention_worker(
    args, 
    worker_id: int,
    slot_num: int,
    kv_vector_byte: int,
    attention_round_task_partition: List[Tuple[int, int, int, int]],
):
    results = []
    output_file = os.path.join(args.test_mid_dir, f"intermediate_{worker_id}.txt")
    for block_size, access_block_num, round_id, simulation_seed in attention_round_task_partition:
        cmd = (
            "{exec_path} {config_path} {test_type} {slot_num} {kv_vector_byte} {block_size} {access_block_num} {simulation_seed} "
            "> {output_file}"
        ).format(
            exec_path=args.exec_path,
            config_path=args.config_path,
            test_type="attention",
            slot_num=slot_num,
            kv_vector_byte=kv_vector_byte,
            block_size=block_size,
            access_block_num=access_block_num,
            simulation_seed=simulation_seed,
            output_file=output_file
        )
        os.system(cmd)

        latency = 0.
        bandwidth = 0.
        util = 0.
        with open(output_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "Latency" in line:
                    latency = float(line.split("Latency: ")[1].split(" s")[0])
                elif "BW Util" in line:
                    util = float(line.split("BW Util: ")[1].split("\n")[0])
                elif "BW" in line:
                    bandwidth = float(line.split("BW: ")[1].split(" GB/s")[0])
        results.append((block_size, access_block_num, round_id, simulation_seed, latency, bandwidth, util))

    return results


def build_attention_round_tasks(attention_combination_list, max_workers, round_count=ATTENTION_ROUND_COUNT):
    base_worker_count = min(len(attention_combination_list), max_workers)
    attention_combination_list_partitions = partition_list(attention_combination_list, max(base_worker_count, 1))

    random.seed(42)
    seed_list = [random.randint(0, 1000000) for _ in range(max(base_worker_count, 1))]

    attention_round_tasks = []
    for worker_id, partition in enumerate(attention_combination_list_partitions):
        random.seed(seed_list[worker_id])
        simulation_seed_list = [random.randint(0, 1000000) for _ in range(round_count)]
        for block_size, access_block_num in partition:
            for round_id, simulation_seed in enumerate(simulation_seed_list):
                attention_round_tasks.append((block_size, access_block_num, round_id, simulation_seed))

    return attention_round_tasks, base_worker_count


def aggregate_attention_round_results(results_list):
    grouped_results = {}
    for worker_results in results_list:
        for block_size, access_block_num, round_id, simulation_seed, latency, bandwidth, util in worker_results:
            grouped_results.setdefault((block_size, access_block_num), []).append(
                (round_id, simulation_seed, latency, bandwidth, util)
            )

    results = {}
    for key, result_list in grouped_results.items():
        result_list.sort(key=lambda result: result[0])
        latency = sum([result[2] for result in result_list]) / len(result_list)
        bandwidth = sum([result[3] for result in result_list]) / len(result_list)
        util = sum([result[4] for result in result_list]) / len(result_list)
        results[key] = {
            "latency": latency,
            "bw": bandwidth,
            "bw_util": util
        }
    return results


def test_attention_access(args):
    args.test_mid_dir = os.path.join(args.mid_dir, "attention")
    args.test_output_dir = os.path.join(args.output_dir, "attention")
    args.exec_path = os.path.join(args.root_dir, "simulator/build/bin/test_dram")
    os.makedirs(args.test_mid_dir, exist_ok=True)
    os.makedirs(args.test_output_dir, exist_ok=True)

    slot_num = 128 * 1024
    kv_vector_byte_list = [512]
    block_size_list = [2**i for i in range(0, 10)]
    context_length_list = [(i*16)*1024//16 for i in [1, 2, 4]]
    attention_combination_list = []
    for block_size in block_size_list:
        for context_length in context_length_list:
            access_block_num = int(math.ceil(context_length / block_size))
            if block_size * access_block_num <= slot_num:
                attention_combination_list.append((block_size, access_block_num))
    attention_round_tasks, base_worker_count = build_attention_round_tasks(
        attention_combination_list,
        args.num_workers,
    )
    num_workers = min(len(attention_round_tasks), args.num_workers)
    attention_round_task_partitions = partition_by_count(attention_round_tasks, max(num_workers, 1))

    all_results = {}
    for kv_vector_byte in kv_vector_byte_list:
        results = {}
        print(
            f"  [RUN] attention kv_vector_byte={kv_vector_byte}: "
            f"{len(attention_combination_list)} configs x {ATTENTION_ROUND_COUNT} rounds = "
            f"{len(attention_round_tasks)} sims, {num_workers} workers"
        )
        with multiprocessing.Pool(max(num_workers, 1), initializer=set_pdeathsig) as pool:
            results_list = pool.starmap(
                attention_worker,
                [
                    (args, worker_id, slot_num, kv_vector_byte, partition)
                    for worker_id, partition in enumerate(attention_round_task_partitions)
                ]
            )
            results = aggregate_attention_round_results(results_list)
        
        with open(os.path.join(args.test_output_dir, f"kv_vector_byte_{kv_vector_byte}.csv"), "w") as f:
            f.write("block_size,access_block_num,latency (s),bw (GB/s),bw_util,bw_util (%)\n")
            for block_size, access_block_num in attention_combination_list:
                f.write(
                    f"{block_size},{access_block_num},"
                    f"{results[(block_size, access_block_num)]['latency']},"
                    f"{results[(block_size, access_block_num)]['bw']},"
                    f"{results[(block_size, access_block_num)]['bw_util']},"
                    f"{results[(block_size, access_block_num)]['bw_util']*100}\n"
                )
        for block_size, access_block_num in attention_combination_list:
            all_results[(kv_vector_byte, block_size, access_block_num)] = results[(block_size, access_block_num)]
    return all_results


def matrix_worker(
    args,
    worker_id: int,
    M: int,
    K: int,
    N: int,
    input_row_major: bool,
    output_row_major: bool,
    weight_row_major: bool,
    element_size: int,
    dataflow: int,
    matrix_combination_list_partition: List[Tuple[int, int, int]]
):
    results = {}
    output_file = os.path.join(args.test_mid_dir, f"intermediate_{worker_id}.txt")
    for tM, tK, tN in matrix_combination_list_partition:
        cmd = (
            "{exec_path} {config_path} {test_type} "
            "{M} {K} {N} {tM} {tK} {tN} "
            "{input_row_major} {output_row_major} {weight_row_major} "
            "{element_size} {dataflow} "
            "> {output_file}"
        ).format(
            exec_path=args.exec_path,
            config_path=args.config_path,
            test_type="matrix",
            M=M,
            K=K,
            N=N,
            tM=tM,
            tK=tK,
            tN=tN,
            input_row_major=input_row_major,
            output_row_major=output_row_major,
            weight_row_major=weight_row_major,
            element_size=element_size,
            dataflow=dataflow,
            output_file=output_file
        )
        os.system(cmd)

        latency = 0.
        bandwidth = 0.
        util = 0.
        with open(output_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "Latency" in line:
                    latency = float(line.split("Latency: ")[1].split(" s")[0])
                elif "BW Util" in line:
                    util = float(line.split("BW Util: ")[1].split("\n")[0])
                elif "BW" in line:
                    bandwidth = float(line.split("BW: ")[1].split(" GB/s")[0])
        
        results[(tM, tK, tN)] = {
            "latency": latency,
            "bw": bandwidth,
            "bw_util": util
        }

    return results


def test_matrix_access(args):
    args.test_mid_dir = os.path.join(args.mid_dir, "matrix")
    args.test_output_dir = os.path.join(args.output_dir, "matrix")
    args.exec_path = os.path.join(args.root_dir, "simulator/build/bin/test_dram")
    os.makedirs(args.test_mid_dir, exist_ok=True)
    os.makedirs(args.test_output_dir, exist_ok=True)

    M = args.M
    K = args.K
    N = args.N
    tM_list = args.tM_list if args.tM_list is not None else get_factors(M, min(M, 16))
    tK_list = args.tK_list if args.tK_list is not None else get_factors(K, min(K, 128))
    tN_list = args.tN_list if args.tN_list is not None else get_factors(N, min(N, 128))
    input_row_majors = [1]
    output_row_majors = [1]
    weight_row_majors = [0]
    matrix_combination_list = list(itertools.product(tM_list, tK_list, tN_list))
    num_workers = min(len(matrix_combination_list), args.num_workers)
    matrix_combination_list_partitions = partition_list(matrix_combination_list, max(num_workers, 1))
    dataflow_name_list = ["os"]
    element_size_list = [2]
    
    all_results = {}
    for dataflow, dataflow_name in enumerate(dataflow_name_list):
        for element_size in element_size_list:
            results = {}
            for input_row_major in input_row_majors:
                for output_row_major in output_row_majors:
                    for weight_row_major in weight_row_majors:
                        print(
                            f"  [RUN] matrix {M}x{K}x{N}: "
                            f"{len(matrix_combination_list)} tiling sims, {num_workers} workers"
                        )
                        with multiprocessing.Pool(max(num_workers, 1), initializer=set_pdeathsig) as pool:
                            results_list = pool.starmap(
                                matrix_worker,
                                [
                                    (
                                        args, worker_id,
                                        M, K, N,
                                        input_row_major, output_row_major, weight_row_major,
                                        element_size, dataflow,
                                        partition
                                    )
                                    for worker_id, partition in enumerate(matrix_combination_list_partitions)
                                ]
                            )
                            tmp_results = {k: v for sub_results in results_list for k, v in sub_results.items()}
                            results[(input_row_major, output_row_major, weight_row_major)] = tmp_results
                
            with open(
                os.path.join(args.test_output_dir,
                f"{dataflow_name}_{element_size}byte_{M}x{K}x{N}.csv"),
                "w"
            ) as f:
                f.write("input_row_major,output_row_major,weight_row_major,tM,tK,tN,latency (s),bw (GB/s),bw_util,bw_util (%)\n")
                for input_row_major in input_row_majors:
                    for output_row_major in output_row_majors:
                        for weight_row_major in weight_row_majors:
                            for tM in tM_list:
                                for tK, tN in itertools.product(tK_list, tN_list):
                                    f.write(
                                        f"{input_row_major},{output_row_major},{weight_row_major},{tM},{tK},{tN},"
                                        f"{results[(input_row_major, output_row_major, weight_row_major)][(tM, tK, tN)]['latency']},"
                                        f"{results[(input_row_major, output_row_major, weight_row_major)][(tM, tK, tN)]['bw']},"
                                        f"{results[(input_row_major, output_row_major, weight_row_major)][(tM, tK, tN)]['bw_util']},"
                                        f"{results[(input_row_major, output_row_major, weight_row_major)][(tM, tK, tN)]['bw_util']*100}\n"
                                    )
                            f.write("\n")

            for input_row_major in input_row_majors:
                for output_row_major in output_row_majors:
                    for weight_row_major in weight_row_majors:
                        for tM in tM_list:
                            for tK, tN in itertools.product(tK_list, tN_list):
                                all_results[(dataflow_name, element_size, input_row_major, output_row_major, weight_row_major, tM, tK, tN)] = results[(input_row_major, output_row_major, weight_row_major)][(tM, tK, tN)]
    return all_results


def test_full_model(args):
    test_mid_dir = os.path.join(args.mid_dir, f"{args.model_name}_bs{args.batch_size}")
    test_output_dir = os.path.join(args.output_dir, f"{args.model_name}_bs{args.batch_size}")
    args.exec_path = os.path.join(args.root_dir, "simulator/build/bin/test_dram")
    os.makedirs(test_mid_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    model_config = get_model_config_from_hf(
        name=args.model_name,
        path=args.model_config_path
    )
    parallel_config = ParallelConfig(
        tp_size=8, ep_size=8
    )
    attention_block, ffn_moe_block = get_layer_operator_list(
        model_config=model_config,
        parallel_config=parallel_config,
        skip_communication=True
    )
    set_shape(
        args,
        model_config=model_config,
        parallel_config=parallel_config,
        attention_block=attention_block,
        ffn_moe_block=ffn_moe_block,
    )
    
    per_operator_performance = []
    for operator in attention_block + ffn_moe_block:
        args.test_mid_dir = os.path.join(test_mid_dir, f"{operator.name}")
        args.test_output_dir = os.path.join(test_output_dir, f"{operator.name}")
        os.makedirs(args.test_mid_dir, exist_ok=True)
        os.makedirs(args.test_output_dir, exist_ok=True)

        if operator.op_type == OperatorType.GEMM:
            M = operator.M
            K = operator.K
            N = operator.N
            tM_list = get_factors(M, min(M, 16))
            tK_list = get_factors(K, min(K, 128))
            tN_list = get_factors(N, min(N, 128))
            if len(tM_list) == 0:
                tM_list = get_factors(M)
            if len(tK_list) == 0:
                tK_list = get_factors(K)
            if len(tN_list) == 0:
                tN_list = get_factors(N)
            
            input_row_major = 1
            output_row_major = 1
            weight_row_major = 0
            dataflow_name_list = ["os"]

            all_results = {}
            for dataflow, dataflow_name in enumerate(dataflow_name_list):
                matrix_combination_list = []
                for tM in tM_list:
                    for tK in tK_list:
                        for tN in tN_list:
                            if args.element_size * (tM*tK + tK*tN + tM*tN) <= args.buffer_size / 2:
                                matrix_combination_list.append((tM, tK, tN))
                num_workers = min(len(matrix_combination_list), args.num_workers)
                matrix_combination_list_partitions = partition_list(matrix_combination_list, max(num_workers, 1))

                print(
                    f"    [RUN] {operator.name}: "
                    f"{len(matrix_combination_list)} tiling sims, {num_workers} workers"
                )
                with multiprocessing.Pool(max(num_workers, 1), initializer=set_pdeathsig) as pool:
                    results_list = pool.starmap(
                        matrix_worker,
                        [
                            (
                                args, worker_id,
                                M, K, N,
                                input_row_major, output_row_major, weight_row_major,
                                args.element_size, dataflow,
                                partition
                            )
                            for worker_id, partition in enumerate(matrix_combination_list_partitions)
                        ]
                    )
                    results = {k: v for sub_results in results_list for k, v in sub_results.items()}
                with open(
                    os.path.join(args.test_output_dir,
                    f"{dataflow_name}_{args.element_size}byte_{M}x{K}x{N}.csv"),
                    "w"
                ) as f:
                    f.write("tM,tK,tN,latency (s),bw (GB/s),bw_util,bw_util (%)\n")
                    for tM, tK, tN in matrix_combination_list:
                        f.write(
                            f"{tM},{tK},{tN},"
                            f"{results[(tM, tK, tN)]['latency']},"
                            f"{results[(tM, tK, tN)]['bw']},"
                            f"{results[(tM, tK, tN)]['bw_util']},"
                            f"{results[(tM, tK, tN)]['bw_util']*100}\n"
                        )

                all_results[dataflow_name] = results
            
            opt_latency = 1919810
            opt_bw = -1
            opt_bw_util = -1
            for dataflow_name, results in all_results.items():
                for _, result in results.items():
                    latency = result['latency']
                    bw = result['bw']
                    bw_util = result['bw_util']
                    if latency < opt_latency:
                        opt_latency = latency
                        opt_bw = bw
                        opt_bw_util = bw_util
            
            per_operator_performance.append({
                "name": operator.name,
                "latency": opt_latency * operator.B,
                "bw": opt_bw,
                "bw_util": opt_bw_util
            })
        elif operator.op_type == OperatorType.ATTENTION:
            slot_num = 128 * 1024

            if operator.is_mla:
                kv_vector_byte = operator.head_dim * args.element_size
            else:
                kv_vector_byte = operator.head_dim * 2 * args.element_size
            
            context_length = sum(operator.context_length)
            access_block_num = int(math.ceil(context_length / args.block_size))
            attention_combination_list = [(args.block_size, access_block_num)]

            results = {}
            attention_round_tasks, base_worker_count = build_attention_round_tasks(
                attention_combination_list,
                args.num_workers,
            )
            num_workers = min(len(attention_round_tasks), args.num_workers)
            attention_round_task_partitions = partition_by_count(attention_round_tasks, max(num_workers, 1))
            print(
                f"    [RUN] {operator.name}: "
                f"{len(attention_combination_list)} attention configs x {ATTENTION_ROUND_COUNT} rounds = "
                f"{len(attention_round_tasks)} sims, {num_workers} workers"
            )
            with multiprocessing.Pool(max(num_workers, 1), initializer=set_pdeathsig) as pool:
                results_list = pool.starmap(
                    attention_worker,
                    [
                        (args, worker_id, slot_num, kv_vector_byte, partition)
                        for worker_id, partition in enumerate(attention_round_task_partitions)
                    ]
                )
                results = aggregate_attention_round_results(results_list)

            with open(os.path.join(args.test_output_dir, f"kv_vector_byte_{kv_vector_byte}.csv"), "w") as f:
                f.write("block_size,access_block_num,latency (s),bw (GB/s),bw_util,bw_util (%)\n")
                for block_size, access_block_num in attention_combination_list:
                    f.write(
                        f"{block_size},{access_block_num},"
                        f"{results[(block_size, access_block_num)]['latency']},"
                        f"{results[(block_size, access_block_num)]['bw']},"
                        f"{results[(block_size, access_block_num)]['bw_util']},"
                        f"{results[(block_size, access_block_num)]['bw_util']*100}\n"
                    )

            opt_latency = 1919810
            opt_bw = -1
            opt_bw_util = -1
            for block_size, access_block_num in attention_combination_list:
                latency = results[(block_size, access_block_num)]['latency']
                bw = results[(block_size, access_block_num)]['bw']
                bw_util = results[(block_size, access_block_num)]['bw_util']
                if latency < opt_latency:
                    opt_latency = latency
                    opt_bw = bw
                    opt_bw_util = bw_util
            
            per_operator_performance.append({
                "name": operator.name,
                "latency": opt_latency,
                "bw": opt_bw,
                "bw_util": opt_bw_util
            })

    total_latency = sum([perf['latency'] for perf in per_operator_performance])
    total_bw = sum([perf['bw']*perf['latency']/total_latency for perf in per_operator_performance])
    total_bw_util = sum([perf['bw_util']*perf['latency']/total_latency for perf in per_operator_performance])
    with open(os.path.join(test_output_dir, "e2e_performance.csv"), "w") as f:
        f.write("name,latency (s),bw (GB/s),bw_util,bw_util (%)\n")
        for perf in per_operator_performance:
            f.write(f"{perf['name']},{perf['latency']},{perf['bw']},{perf['bw_util']},{perf['bw_util']*100}\n")
        f.write(f"total,{total_latency},{total_bw},{total_bw_util},{total_bw_util*100}\n")


test_mapping = {
    "attention": test_attention_access,
    "matrix": test_matrix_access,
    "model": test_full_model,
}


if __name__ == "__main__":
    args = parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args.root_dir = os.path.relpath(project_root, os.getcwd())

    args.mid_dir = os.path.join(args.root_dir, args.mid_dir)
    args.output_dir = os.path.join(args.root_dir, args.output_dir)

    os.makedirs(args.mid_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.test_type == "attention":
        test_attention_access(args)
    elif args.test_type == "matrix":
        args.M = 64
        args.K = 2048
        args.N = 2048
        args.tM_list = [64]
        test_matrix_access(args)
    elif args.test_type == "model_dense":
        args.model_name = "llama3_70b"
        args.model_config_path = "configs/models/llama3_70b.json"
        args.batch_size = 8
        args.context_length = 16*1024
        args.element_size = 2
        args.block_size = 64
        args.buffer_size = 1*1024*1024
        test_full_model(args)
    elif args.test_type == "model_moe":
        args.model_name = "qwen3_235b_a22b"
        args.model_config_path = "configs/models/qwen3_235b_a22b.json"
        args.batch_size = 8
        args.context_length = 16*1024
        args.element_size = 1
        args.block_size = 64
        args.buffer_size = 1*1024*1024
        test_full_model(args)
    else:
        raise ValueError(f"Invalid test type: {args.test_type}")
