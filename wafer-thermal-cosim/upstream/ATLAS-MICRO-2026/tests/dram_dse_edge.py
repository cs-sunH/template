import argparse
import csv
import yaml
import sys
import os
import math
import re
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dram_test import *
from ae_utils import (
    EDGE_FC_TILE_ORDER,
    EDGE_QK_TILE_ORDER,
    EDGE_SV_TILE_ORDER,
    add_dram_percent_colorbar,
    apply_paper_style,
    draw_dram_heatmap_pct,
    ensure_dir,
    format_tile,
    save_basic_figure,
    to_float,
    write_csv_dicts,
)


EDGE_DRAM_LINE_STYLES = [
    {"color": "#2E75B6", "marker": "s", "markerfacecolor": "white"},
    {"color": "#C55A11", "marker": "^", "markerfacecolor": "white"},
    {"color": "#7C7C7C", "marker": "o", "markerfacecolor": "white"},
    {"color": "#FFC000", "marker": "D", "markerfacecolor": "white"},
]


def _mp_ctx():
    return multiprocessing.get_context("spawn")


column_bit_dict = {
    "HBDRAM_2Gb_512pin_2KB_edge": 5,
    "HBDRAM_2Gb_512pin_4KB_edge": 6,
    "HBDRAM_2Gb_512pin_8KB_edge": 7,
    "HBDRAM_2Gb_512pin_16KB_edge": 8,

    "HBDRAM_1Gb_256pin_2KB_edge": 6,
    "HBDRAM_1Gb_256pin_4KB_edge": 7,
    "HBDRAM_1Gb_256pin_8KB_edge": 8,
    "HBDRAM_1Gb_256pin_16KB_edge": 9,

    "HBDRAM_512Mb_128pin_2KB_edge": 7,
    "HBDRAM_512Mb_128pin_4KB_edge": 8,
    "HBDRAM_512Mb_128pin_8KB_edge": 9,
    "HBDRAM_512Mb_128pin_16KB_edge": 10,

    "HBDRAM_256Mb_64pin_2KB_edge": 8,
    "HBDRAM_256Mb_64pin_4KB_edge": 9,
    "HBDRAM_256Mb_64pin_8KB_edge": 10,
    "HBDRAM_256Mb_64pin_16KB_edge": 11,
}


channel_num_dict = {
    "HBDRAM_2Gb_512pin_2KB_edge": 1,
    "HBDRAM_2Gb_512pin_4KB_edge": 1,
    "HBDRAM_2Gb_512pin_8KB_edge": 1,
    "HBDRAM_2Gb_512pin_16KB_edge": 1,

    "HBDRAM_1Gb_256pin_2KB_edge": 2,
    "HBDRAM_1Gb_256pin_4KB_edge": 2,
    "HBDRAM_1Gb_256pin_8KB_edge": 2,
    "HBDRAM_1Gb_256pin_16KB_edge": 2,

    "HBDRAM_512Mb_128pin_2KB_edge": 4,
    "HBDRAM_512Mb_128pin_4KB_edge": 4,
    "HBDRAM_512Mb_128pin_8KB_edge": 4,
    "HBDRAM_512Mb_128pin_16KB_edge": 4,

    "HBDRAM_256Mb_64pin_2KB_edge": 8,
    "HBDRAM_256Mb_64pin_4KB_edge": 8,
    "HBDRAM_256Mb_64pin_8KB_edge": 8,
    "HBDRAM_256Mb_64pin_16KB_edge": 8,
}


logical_row_size_dict = {
    "HBDRAM_2Gb_512pin_2KB_edge": 2,
    "HBDRAM_2Gb_512pin_4KB_edge": 4,
    "HBDRAM_2Gb_512pin_8KB_edge": 8,
    "HBDRAM_2Gb_512pin_16KB_edge": 16,

    "HBDRAM_1Gb_256pin_2KB_edge": 2,
    "HBDRAM_1Gb_256pin_4KB_edge": 4,
    "HBDRAM_1Gb_256pin_8KB_edge": 8,
    "HBDRAM_1Gb_256pin_16KB_edge": 16,

    "HBDRAM_512Mb_128pin_2KB_edge": 2,
    "HBDRAM_512Mb_128pin_4KB_edge": 4,
    "HBDRAM_512Mb_128pin_8KB_edge": 8,
    "HBDRAM_512Mb_128pin_16KB_edge": 16,

    "HBDRAM_256Mb_64pin_2KB_edge": 2,
    "HBDRAM_256Mb_64pin_4KB_edge": 4,
    "HBDRAM_256Mb_64pin_8KB_edge": 8,
    "HBDRAM_256Mb_64pin_16KB_edge": 16,
}


timing_dict = {k: "HBDRAM_400Mbps" for k in column_bit_dict}

prefetch_dict = {k: 1 for k in column_bit_dict}


SYSTEM_CHANNEL_NUM = 8


chip_config = {
    "architecture": {
        "frequency": 600,
        "core_num": 16,
        "core": {
            "controller": {
                "power": 1.0,
                "area": 1.0,
            },
            "matrix": {
                "mac_num": 128,
                "power": 1.1,
                "area": 1.1,
            },
            "vector": {
                "vec_num": 8,
                "power": 1.2,
                "area": 1.2,
            },
            "buffer": {
                "buffer_size": 38,
                "read_bw": 8192,
                "write_bw": 8192,
                "power": 1.3,
                "area": 1.3,
            }
        },
        "dram": {
            "config_path": "",
            "power": 1.4,
            "area": 1.4,
        },
    }
}


dram_config = {
    "Frontend": {
        "impl": "HBFrontend",
        "clock_ratio": 1
    },
    "MemorySystem": {
        "impl": "GenericDRAM",
        "clock_ratio": 1,
        "DRAM": {
            "impl": "HBDRAM",
            "org": {
                "preset": "HBDRAM_256Mb_64pin_8KB_edge",
                "channel": 8,
                "internal_prefetch_size": 1,
            },
            "timing": {
                "preset": "HBDRAM_400Mbps"
            }
        },
        "Controller": {
            "impl": "Generic",
            "Scheduler": {
                "impl": "FRFCFS"
            },
            "RefreshManager": {
                "impl": "AllBank"
            },
            "RowPolicy": {
                "impl": "OpenRowPolicy"
            },
            "plugins": None
        },
        "AddrMapper": {
            "impl": "OneLevelInterleave",
            "channel_lowest_bit": 0
        }
    }
}


def represent_none(self, _):
    return self.represent_scalar('tag:yaml.org,2002:null', '')
yaml.add_representer(type(None), represent_none)


def _write_matrix_summary_csv(summary_csv_path, config_output_dirs):
    base, ext = os.path.splitext(summary_csv_path)
    all_csv = {}

    for label, base_dir, bits in config_output_dirs:
        for b in sorted(bits):
            matrix_dir = os.path.join(base_dir, f"interleave_{b}bit", "matrix")
            if not os.path.isdir(matrix_dir):
                continue
            for fname in sorted(os.listdir(matrix_dir)):
                if not fname.endswith('.csv'):
                    continue
                if fname not in all_csv:
                    all_csv[fname] = {"configs": [], "keys_ordered": [], "keys_set": set(), "results": {}}
                entry = all_csv[fname]
                cfg = (label, b)
                entry["configs"].append(cfg)

                with open(os.path.join(matrix_dir, fname), 'r') as f:
                    lines = f.readlines()
                for line in lines[1:]:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) < 9:
                        continue
                    tk = (parts[3], parts[4], parts[5])
                    if tk not in entry["keys_set"]:
                        entry["keys_set"].add(tk)
                        entry["keys_ordered"].append(tk)
                    entry["results"].setdefault(tk, {})[cfg] = (parts[6], parts[7], parts[8])

    metric_defs = [
        ("latency", "latency(s)", 0),
        ("bw", "bw(GB/s)", 1),
        ("bw_util", "bw_util", 2),
    ]
    for metric_key, col_suffix, val_idx in metric_defs:
        with open(f"{base}_{metric_key}{ext}", "w") as out:
            for fname in sorted(all_csv.keys()):
                entry = all_csv[fname]
                out.write(f"# {fname}\n")
                hdr = ["tM", "tK", "tN"]
                for lbl, b in entry["configs"]:
                    prefix = f"{lbl}_" if lbl else ""
                    hdr.append(f"{prefix}bit{b}_{col_suffix}")
                out.write(",".join(hdr) + "\n")
                for tk in entry["keys_ordered"]:
                    row = list(tk)
                    for cfg in entry["configs"]:
                        vals = entry["results"].get(tk, {}).get(cfg)
                        row.append(vals[val_idx] if vals else "")
                    out.write(",".join(row) + "\n")
                out.write("\n")


def _write_model_summary_csv(summary_csv_path, config_output_dirs):
    base, ext = os.path.splitext(summary_csv_path)
    all_models = {}
    labels_ordered = []
    labels_seen = set()

    for label, base_dir, bits in config_output_dirs:
        if label not in labels_seen:
            labels_seen.add(label)
            labels_ordered.append(label)
        for b in sorted(bits):
            interleave_dir = os.path.join(base_dir, f"interleave_{b}bit")
            if not os.path.isdir(interleave_dir):
                continue
            for model_dir_name in sorted(os.listdir(interleave_dir)):
                e2e_path = os.path.join(interleave_dir, model_dir_name, "e2e_performance.csv")
                if not os.path.isfile(e2e_path):
                    continue
                all_models.setdefault(model_dir_name, {})
                all_models[model_dir_name].setdefault(label, {})

                with open(e2e_path, 'r') as f:
                    lines = f.readlines()
                for line in lines[1:]:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) >= 4 and parts[0] == "total":
                        all_models[model_dir_name][label][b] = (parts[1], parts[2], parts[3])
                        break

    metric_defs = [
        ("latency", "total_latency(s)", 0),
        ("bw", "total_bw(GB/s)", 1),
        ("bw_util", "total_bw_util", 2),
    ]
    for metric_key, col_name, val_idx in metric_defs:
        with open(f"{base}_{metric_key}{ext}", "w") as out:
            for model_dir_name in sorted(all_models.keys()):
                entry = all_models[model_dir_name]
                out.write(f"# {model_dir_name}\n")
                out.write(f"preset,interleave_bit,{col_name}\n")
                for label in labels_ordered:
                    if label not in entry:
                        continue
                    bit_data = entry[label]
                    best_bit = None
                    best_latency = float('inf')
                    best_vals = None
                    for b, vals in bit_data.items():
                        try:
                            lat = float(vals[0])
                        except ValueError:
                            continue
                        if lat < best_latency:
                            best_latency = lat
                            best_bit = b
                            best_vals = vals
                    if best_vals is not None:
                        out.write(f"{label},{best_bit},{best_vals[val_idx]}\n")
                out.write("\n")


def _edge_matrix_worker(args, worker_id, work_items):
    results = {}
    output_file = os.path.join(args.test_mid_dir, f"intermediate_{worker_id}.txt")
    for M, K, N, tM, tK, tN in work_items:
        cmd = (
            "{exec_path} {config_path} matrix "
            "{M} {K} {N} {tM} {tK} {tN} "
            "1 1 0 "
            "{element_size} 0 "
            "> {output_file}"
        ).format(
            exec_path=args.exec_path,
            config_path=args.config_path,
            M=M, K=K, N=N, tM=tM, tK=tK, tN=tN,
            element_size=args.element_size,
            output_file=output_file
        )
        os.system(cmd)

        latency = 0.
        bandwidth = 0.
        util = 0.
        with open(output_file, "r") as f:
            for line in f:
                if "Latency" in line:
                    latency = float(line.split("Latency: ")[1].split(" s")[0])
                elif "BW Util" in line:
                    util = float(line.split("BW Util: ")[1].split("\n")[0])
                elif "BW" in line:
                    bandwidth = float(line.split("BW: ")[1].split(" GB/s")[0])
        results[(M, K, N, tM, tK, tN)] = {
            "latency": latency, "bw": bandwidth, "bw_util": util
        }
    return results


def _should_skip_matrix(args):
    tag = f"matrix {args.M}x{args.K}x{args.N}"
    csv_path = os.path.join(
        args.output_dir, "matrix", f"os_2byte_{args.M}x{args.K}x{args.N}.csv")
    if getattr(args, 'use_cache', False) and os.path.isfile(csv_path):
        print(f"  [CACHE] {tag} — {csv_path}")
        return True
    print(f"  [RUN]   {tag}")
    return False


def _should_skip_model(args):
    tag = f"model {args.model_name}_bs{args.batch_size}"
    csv_path = os.path.join(
        args.output_dir, f"{args.model_name}_bs{args.batch_size}",
        "e2e_performance.csv")
    if getattr(args, 'use_cache', False) and os.path.isfile(csv_path):
        print(f"  [CACHE] {tag} — {csv_path}")
        return True
    return False


def set_shape_edge(
    args,
    model_config: ModelConfig,
    attention_block: List[Operator],
    ffn_moe_block: List[Operator],
):
    batch_size = args.batch_size
    context_length = args.context_length

    for operator in attention_block:
        if operator.op_type == OperatorType.GEMM:
            if "attention_qk" in operator.name:
                operator.N = context_length
            elif "attention_sv" in operator.name:
                operator.K = context_length
            operator.M = batch_size

    if model_config.is_moe:
        per_ffn_token_num = math.ceil(
            batch_size * (model_config.moe_top_k + model_config.moe_shared_experts) / model_config.moe_num_experts
        )
        total_ffn_num = math.ceil(
            min(batch_size * (model_config.moe_top_k + model_config.moe_shared_experts), model_config.moe_num_experts)
        )
    else:
        total_ffn_num = 1
        per_ffn_token_num = batch_size

    for operator in ffn_moe_block:
        if operator.op_type == OperatorType.GEMM:
            operator.B = total_ffn_num
            operator.M = per_ffn_token_num


def test_full_model_edge(args):
    print(f"  [RUN] model {args.model_name}_bs{args.batch_size}")
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
        tp_size=1, ep_size=1
    )
    attention_block, ffn_moe_block = get_layer_operator_list(
        model_config=model_config,
        parallel_config=parallel_config,
        skip_communication=True,
        non_fused_attention=False
    )

    edge_attention_block = []
    for operator in attention_block:
        if operator.op_type == OperatorType.GEMM:
            edge_attention_block.append(operator)
        elif operator.op_type == OperatorType.ATTENTION:
            edge_attention_block.append(Operator(
                name="attention_qk",
                op_type=OperatorType.GEMM,
                tp_size=parallel_config.tp_size,
                ep_size=parallel_config.ep_size,
                B=operator.kv_head_num,
                M=operator.kv_group_num,
                K=operator.head_dim,
                N=1,
            ))
            edge_attention_block.append(Operator(
                name="attention_sv",
                op_type=OperatorType.GEMM,
                tp_size=parallel_config.tp_size,
                ep_size=parallel_config.ep_size,
                B=operator.kv_head_num,
                M=operator.kv_group_num,
                K=1,
                N=operator.head_dim,
            ))
    attention_block = edge_attention_block

    set_shape_edge(
        args,
        model_config=model_config,
        attention_block=attention_block,
        ffn_moe_block=ffn_moe_block,
    )

    per_operator_performance = []
    system_channel_num = args.system_channel_num
    core_num = chip_config["architecture"]["core_num"]
    channel_num_factor_list = get_factors(system_channel_num)
    core_num_factor_list = get_factors(core_num)

    all_operators = attention_block + ffn_moe_block
    for op_idx, operator in enumerate(all_operators):
        args.test_mid_dir = os.path.join(test_mid_dir, f"{operator.name}")
        args.test_output_dir = os.path.join(test_output_dir, f"{operator.name}")
        os.makedirs(args.test_mid_dir, exist_ok=True)
        os.makedirs(args.test_output_dir, exist_ok=True)

        if operator.op_type == OperatorType.GEMM:
            sim_to_channel_Bs = {}

            for channel_num_factor in channel_num_factor_list:
                if channel_num_factor > operator.B:
                    continue
                B_per_group = math.ceil(operator.B / channel_num_factor)
                channel_per_group = math.ceil(system_channel_num / channel_num_factor)

                for channel_nK in get_factors(channel_per_group):
                    channel_B = B_per_group
                    channel_K = math.ceil(operator.K / channel_nK)
                    channel_N = math.ceil(operator.N / (channel_per_group // channel_nK))

                    for core_nK in core_num_factor_list:
                        M = operator.M
                        K = math.ceil(channel_K / core_nK)
                        N = math.ceil(channel_N / (core_num // core_nK))

                        tM_list = get_factors(M, min(M, 16))
                        tK_list = get_factors(K, min(K, 16))
                        tN_list = get_factors(N, min(N, 16))
                        if len(tM_list) <= 2    :
                            tM_list = get_factors(M)
                        if len(tK_list) <= 2:
                            tK_list = get_factors(K)
                        if len(tN_list) <= 2:
                            tN_list = get_factors(N)

                        for tM in tM_list:
                            for tK in tK_list:
                                for tN in tN_list:
                                    if args.element_size * (tM*tK + tK*tN + tM*tN) <= args.buffer_size / 2:
                                        key = (M, K, N, tM, tK, tN)
                                        sim_to_channel_Bs.setdefault(key, set()).add(channel_B)

            sim_items = list(sim_to_channel_Bs.keys())

            opt_latency = float('inf')
            opt_bw = -1
            opt_bw_util = -1

            if sim_items:
                num_workers = min(len(sim_items), args.num_workers)
                partitions = partition_list(sim_items, max(num_workers, 1))

                print(f"    [{op_idx+1}/{len(all_operators)}] {operator.name}: "
                      f"{len(sim_items)} sims, {num_workers} workers")
                ctx = _mp_ctx()
                try:
                    with ctx.Pool(max(num_workers, 1), initializer=set_pdeathsig) as pool:
                        results_list = pool.starmap(
                            _edge_matrix_worker,
                            [(args, wid, part) for wid, part in enumerate(partitions)]
                        )
                except Exception as e:
                    print(f"[ERROR] Edge DRAM multiprocessing failed for {operator.name}: {e}")
                    raise

                all_sim_results = {}
                for sub_results in results_list:
                    all_sim_results.update(sub_results)

                for key, result in all_sim_results.items():
                    for channel_B in sim_to_channel_Bs[key]:
                        lat = result['latency'] * channel_B
                        if lat < opt_latency:
                            opt_latency = lat
                            opt_bw = result['bw']
                            opt_bw_util = result['bw_util']

                with open(os.path.join(args.test_output_dir,
                          f"os_{args.element_size}byte_{operator.M}x{operator.K}x{operator.N}.csv"), "w") as f:
                    f.write("M,K,N,tM,tK,tN,channel_B,latency (s),bw (GB/s),bw_util,bw_util (%)\n")
                    for key in sorted(all_sim_results.keys()):
                        M_k, K_k, N_k, tM_k, tK_k, tN_k = key
                        r = all_sim_results[key]
                        for cB in sorted(sim_to_channel_Bs[key]):
                            f.write(f"{M_k},{K_k},{N_k},{tM_k},{tK_k},{tN_k},{cB},"
                                    f"{r['latency']*cB},{r['bw']},{r['bw_util']},{r['bw_util']*100}\n")
            else:
                print(f"    [{op_idx+1}/{len(all_operators)}] {operator.name}: "
                      f"no valid tiling configs")

            per_operator_performance.append({
                "name": operator.name,
                "latency": opt_latency,
                "bw": opt_bw,
                "bw_util": opt_bw_util,
            })
        else:
            assert False, f"Only explore GEMM operators' DRAM access"

    total_latency = sum([perf['latency'] for perf in per_operator_performance])
    total_bw = sum([perf['bw']*perf['latency']/total_latency for perf in per_operator_performance])
    total_bw_util = sum([perf['bw_util']*perf['latency']/total_latency for perf in per_operator_performance])
    e2e_csv_path = os.path.join(test_output_dir, "e2e_performance.csv")
    with open(e2e_csv_path, "w") as f:
        f.write("name,latency (s),bw (GB/s),bw_util,bw_util (%)\n")
        for perf in per_operator_performance:
            f.write(f"{perf['name']},{perf['latency']},{perf['bw']},{perf['bw_util']},{perf['bw_util']*100}\n")
        f.write(f"total,{total_latency},{total_bw},{total_bw_util},{total_bw_util*100}\n")
    print(f"  [DONE] model {args.model_name}_bs{args.batch_size} -> {e2e_csv_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="all", choices=["01_channel_interleaving", "02_io_organization", "03_logical_row_size", "04_full_space", "interleave", "io", "row", "full", "fig20", "all"])
    parser.add_argument("--action", type=str, default="all", choices=["run", "collect", "plot", "all"])
    parser.add_argument("--mid-dir", type=str, default="results/dram_dse_edge/mid_results")
    parser.add_argument("--output-dir", type=str, default="results/dram_dse_edge/output_results")
    parser.add_argument("--num-workers", type=int, default=int(multiprocessing.cpu_count()*0.8))
    parser.add_argument("--system-channel-num", type=int, default=SYSTEM_CHANNEL_NUM)
    parser.add_argument("--use-cache", action="store_true", help="Skip simulations whose output files already exist")
    return parser.parse_args()


def dse_interleave(args):
    print("=" * 60)
    print("[DSE] dse_interleave")
    print("=" * 60)
    org_preset = "HBDRAM_512Mb_128pin_4KB_edge"
    timing_preset = timing_dict[org_preset]
    column_bit_count = column_bit_dict[org_preset]
    channel_num = channel_num_dict[org_preset]

    mid_root_dir = os.path.join(args.mid_dir, f"01_channel_interleaving")
    output_root_dir = os.path.join(args.output_dir, f"01_channel_interleaving")
    os.makedirs(mid_root_dir, exist_ok=True)
    os.makedirs(output_root_dir, exist_ok=True)

    config_dir = os.path.join(mid_root_dir, f"configs")
    os.makedirs(config_dir, exist_ok=True)
    for pb, b in enumerate([1, 3, 5, 7]):
        print(f"[{pb+1}/{len([1, 3, 5, 7])}] {org_preset}, interleave bit={b}")
        dram_config_file = os.path.join(config_dir, f"dram_config_{b}bit.yaml")
        dram_config["MemorySystem"]["DRAM"]["org"]["preset"] = org_preset
        dram_config["MemorySystem"]["DRAM"]["org"]["channel"] = channel_num
        dram_config["MemorySystem"]["DRAM"]["timing"]["preset"] = timing_preset
        dram_config["MemorySystem"]["AddrMapper"]["impl"] = "OneLevelInterleave"
        dram_config["MemorySystem"]["AddrMapper"]["channel_lowest_bit"] = b
        with open(dram_config_file, "w") as f:
            yaml.dump(dram_config, f, sort_keys=False, default_flow_style=False)

        chip_config_file = os.path.join(config_dir, f"chip_config_{b}bit.yaml")
        chip_config["architecture"]["dram"]["config_path"] = dram_config_file
        with open(chip_config_file, "w") as f:
            yaml.dump(chip_config, f, sort_keys=False, default_flow_style=False)

        args.mid_dir = os.path.join(mid_root_dir, f"worker_outputs")
        args.output_dir = os.path.join(output_root_dir, f"interleave_{b}bit")
        args.config_path = chip_config_file
        os.makedirs(args.mid_dir, exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)

        for test_type in ["matrix"]:
            args.test_type = test_type
            if test_type == "matrix":
                args.M = 8
                args.K = 4096
                args.N = 32
                args.tM_list = [8]
                args.tK_list = [128, 256, 512, 1024, 2048]
                args.tN_list = [4, 32]
                args.test_type = "matrix"
                if not _should_skip_matrix(args):
                    test_mapping[test_type](args)

                args.M = 8
                args.K = 128
                args.N = 128
                args.tM_list = [8]
                args.tK_list = [8, 16, 32, 64, 128]
                args.tN_list = [16, 64]
                args.test_type = "matrix"
                if not _should_skip_matrix(args):
                    test_mapping[test_type](args)
                
                args.M = 8
                args.K = 2048
                args.N = 8
                args.tM_list = [8]
                args.tK_list = [128, 256, 512, 1024, 2048]
                args.tN_list = [2, 8]
                args.test_type = "matrix"
                if not _should_skip_matrix(args):
                    test_mapping[test_type](args)
            else:
                test_mapping[test_type](args)

    summary_path = os.path.join(output_root_dir, "summary.csv")
    _write_matrix_summary_csv(
        summary_path,
        [("", output_root_dir, list(range(column_bit_count + 1)))]
    )
    print(f"[DSE] dse_interleave done -> {summary_path.replace('.csv', '_{{latency,bw,bw_util}}.csv')}")


def dse_io_organization(args):
    print("=" * 60)
    print("[DSE] dse_io_organization")
    print("=" * 60)
    mid_root_dir = os.path.join(args.mid_dir, f"02_io_organization")
    output_root_dir = os.path.join(args.output_dir, f"02_io_organization")
    os.makedirs(mid_root_dir, exist_ok=True)
    os.makedirs(output_root_dir, exist_ok=True)

    org_presets = [
        "HBDRAM_2Gb_512pin_4KB_edge",
        "HBDRAM_1Gb_256pin_4KB_edge",
        "HBDRAM_512Mb_128pin_4KB_edge",
        "HBDRAM_256Mb_64pin_4KB_edge",
    ]

    preset_summary_bits = {}
    for pi, org_preset in enumerate(org_presets):
        print(f"[{pi+1}/{len(org_presets)}] {org_preset}")
        timing_preset = timing_dict[org_preset]
        column_bit_count = column_bit_dict[org_preset]
        channel_num = channel_num_dict[org_preset]

        cur_org_mid_dir = os.path.join(mid_root_dir, f"channel_num_{channel_num}")
        cur_org_output_dir = os.path.join(output_root_dir, f"channel_num_{channel_num}")
        os.makedirs(cur_org_mid_dir, exist_ok=True)
        os.makedirs(cur_org_output_dir, exist_ok=True)

        config_dir = os.path.join(cur_org_mid_dir, f"configs")
        os.makedirs(config_dir, exist_ok=True)

        b = min(column_bit_count, 0)
        preset_summary_bits[org_preset] = b
        dram_config_file = os.path.join(config_dir, f"dram_config.yaml")
        dram_config["MemorySystem"]["DRAM"]["org"]["preset"] = org_preset
        dram_config["MemorySystem"]["DRAM"]["org"]["channel"] = channel_num
        dram_config["MemorySystem"]["DRAM"]["timing"]["preset"] = timing_preset
        dram_config["MemorySystem"]["AddrMapper"]["impl"] = "OneLevelInterleave"
        dram_config["MemorySystem"]["AddrMapper"]["channel_lowest_bit"] = b
        with open(dram_config_file, "w") as f:
            yaml.dump(dram_config, f, sort_keys=False, default_flow_style=False)

        chip_config_file = os.path.join(config_dir, f"chip_config.yaml")
        chip_config["architecture"]["dram"]["config_path"] = dram_config_file
        with open(chip_config_file, "w") as f:
            yaml.dump(chip_config, f, sort_keys=False, default_flow_style=False)

        args.mid_dir = os.path.join(cur_org_mid_dir, f"worker_outputs")
        args.output_dir = os.path.join(cur_org_output_dir, f"final_outputs")
        args.config_path = chip_config_file
        os.makedirs(args.mid_dir, exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)

        for test_type in ["matrix"]:
            args.test_type = test_type
            if test_type == "matrix":
                args.M = 8
                args.K = 4096
                args.N = 32
                args.tM_list = [8]
                args.tK_list = [128, 256, 512, 1024, 2048]
                args.tN_list = [4, 32]
                args.test_type = "matrix"
                if not _should_skip_matrix(args):
                    test_mapping[test_type](args)

                args.M = 8
                args.K = 128
                args.N = 128
                args.tM_list = [8]
                args.tK_list = [8, 16, 32, 64, 128]
                args.tN_list = [16, 64]
                args.test_type = "matrix"
                if not _should_skip_matrix(args):
                    test_mapping[test_type](args)
                
                args.M = 8
                args.K = 2048
                args.N = 8
                args.tM_list = [8]
                args.tK_list = [128, 256, 512, 1024, 2048]
                args.tN_list = [2, 8]
                args.test_type = "matrix"
                if not _should_skip_matrix(args):
                    test_mapping[test_type](args)
            else:
                test_mapping[test_type](args)

    summary_path = os.path.join(output_root_dir, "summary.csv")
    _write_matrix_summary_csv(
        summary_path,
        [
            (org_preset, os.path.join(output_root_dir, f"channel_num_{channel_num_dict[org_preset]}"),
             [preset_summary_bits[org_preset]])
            for org_preset in org_presets
        ]
    )
    print(f"[DSE] dse_io_organization done -> {summary_path.replace('.csv', '_{{latency,bw,bw_util}}.csv')}")


def dse_logical_row_size(args):
    print("=" * 60)
    print("[DSE] dse_logical_row_size")
    print("=" * 60)
    mid_root_dir = os.path.join(args.mid_dir, f"03_logical_row_size")
    output_root_dir = os.path.join(args.output_dir, f"03_logical_row_size")
    os.makedirs(mid_root_dir, exist_ok=True)
    os.makedirs(output_root_dir, exist_ok=True)

    org_presets = [
        "HBDRAM_512Mb_128pin_2KB_edge",
        "HBDRAM_512Mb_128pin_4KB_edge",
        "HBDRAM_512Mb_128pin_8KB_edge",
        "HBDRAM_512Mb_128pin_16KB_edge",
    ]

    preset_summary_bits = {}
    for pi, org_preset in enumerate(org_presets):
        print(f"[{pi+1}/{len(org_presets)}] {org_preset}")
        timing_preset = timing_dict[org_preset]
        column_bit_count = column_bit_dict[org_preset]
        channel_num = channel_num_dict[org_preset]
        logical_row_size = logical_row_size_dict[org_preset]

        cur_org_mid_dir = os.path.join(mid_root_dir, f"logical_row_size_{logical_row_size}KB")
        cur_org_output_dir = os.path.join(output_root_dir, f"logical_row_size_{logical_row_size}KB")
        os.makedirs(cur_org_mid_dir, exist_ok=True)
        os.makedirs(cur_org_output_dir, exist_ok=True)

        config_dir = os.path.join(cur_org_mid_dir, f"configs")
        os.makedirs(config_dir, exist_ok=True)

        b = min(column_bit_count, 0)
        preset_summary_bits[org_preset] = b
        dram_config_file = os.path.join(config_dir, f"dram_config.yaml")
        dram_config["MemorySystem"]["DRAM"]["org"]["preset"] = org_preset
        dram_config["MemorySystem"]["DRAM"]["org"]["channel"] = channel_num
        dram_config["MemorySystem"]["DRAM"]["timing"]["preset"] = timing_preset
        dram_config["MemorySystem"]["AddrMapper"]["impl"] = "OneLevelInterleave"
        dram_config["MemorySystem"]["AddrMapper"]["channel_lowest_bit"] = b
        with open(dram_config_file, "w") as f:
            yaml.dump(dram_config, f, sort_keys=False, default_flow_style=False)

        chip_config_file = os.path.join(config_dir, f"chip_config.yaml")
        chip_config["architecture"]["dram"]["config_path"] = dram_config_file
        with open(chip_config_file, "w") as f:
            yaml.dump(chip_config, f, sort_keys=False, default_flow_style=False)

        args.mid_dir = os.path.join(cur_org_mid_dir, f"worker_outputs")
        args.output_dir = os.path.join(cur_org_output_dir, f"final_outputs")
        args.config_path = chip_config_file
        os.makedirs(args.mid_dir, exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)

        for test_type in ["matrix"]:
            args.test_type = test_type
            if test_type == "matrix":
                args.M = 8
                args.K = 4096
                args.N = 32
                args.tM_list = [8]
                args.tK_list = [128, 256, 512, 1024, 2048]
                args.tN_list = [4, 32]
                args.test_type = "matrix"
                if not _should_skip_matrix(args):
                    test_mapping[test_type](args)

                args.M = 8
                args.K = 128
                args.N = 128
                args.tM_list = [8]
                args.tK_list = [8, 16, 32, 64, 128]
                args.tN_list = [16, 64]
                args.test_type = "matrix"
                if not _should_skip_matrix(args):
                    test_mapping[test_type](args)
                
                args.M = 8
                args.K = 2048
                args.N = 8
                args.tM_list = [8]
                args.tK_list = [128, 256, 512, 1024, 2048]
                args.tN_list = [2, 8]
                args.test_type = "matrix"
                if not _should_skip_matrix(args):
                    test_mapping[test_type](args)
            else:
                test_mapping[test_type](args)

    summary_path = os.path.join(output_root_dir, "summary.csv")
    _write_matrix_summary_csv(
        summary_path,
        [
            (org_preset, os.path.join(output_root_dir, f"logical_row_size_{logical_row_size_dict[org_preset]}KB"),
             [preset_summary_bits[org_preset]])
            for org_preset in org_presets
        ]
    )
    print(f"[DSE] dse_logical_row_size done -> {summary_path.replace('.csv', '_{{latency,bw,bw_util}}.csv')}")


def dse_full_space(args):
    print("=" * 60)
    print("[DSE] dse_full_space")
    print("=" * 60)
    mid_root_dir = os.path.join(args.mid_dir, f"04_full_space")
    output_root_dir = os.path.join(args.output_dir, f"04_full_space")
    os.makedirs(mid_root_dir, exist_ok=True)
    os.makedirs(output_root_dir, exist_ok=True)

    org_presets = [
        "HBDRAM_2Gb_512pin_2KB_edge",
        "HBDRAM_2Gb_512pin_4KB_edge",
        "HBDRAM_2Gb_512pin_8KB_edge",
        "HBDRAM_2Gb_512pin_16KB_edge",

        "HBDRAM_1Gb_256pin_2KB_edge",
        "HBDRAM_1Gb_256pin_4KB_edge",
        "HBDRAM_1Gb_256pin_8KB_edge",
        "HBDRAM_1Gb_256pin_16KB_edge",

        "HBDRAM_512Mb_128pin_2KB_edge",
        "HBDRAM_512Mb_128pin_4KB_edge",
        "HBDRAM_512Mb_128pin_8KB_edge",
        "HBDRAM_512Mb_128pin_16KB_edge",

        "HBDRAM_256Mb_64pin_2KB_edge",
        "HBDRAM_256Mb_64pin_4KB_edge",
        "HBDRAM_256Mb_64pin_8KB_edge",
        "HBDRAM_256Mb_64pin_16KB_edge",
    ]

    preset_summary_bits = {}
    for pi, org_preset in enumerate(org_presets):
        print(f"[{pi+1}/{len(org_presets)}] {org_preset}")
        timing_preset = timing_dict[org_preset]
        column_bit_count = column_bit_dict[org_preset]
        channel_num = channel_num_dict[org_preset]
        prefetch_size = prefetch_dict[org_preset]
        logical_row_size = logical_row_size_dict[org_preset]

        cur_org_mid_dir = os.path.join(mid_root_dir, f"{channel_num}channels_{logical_row_size}KB")
        cur_org_output_dir = os.path.join(output_root_dir, f"{channel_num}channels_{logical_row_size}KB")
        os.makedirs(cur_org_mid_dir, exist_ok=True)
        os.makedirs(cur_org_output_dir, exist_ok=True)

        config_dir = os.path.join(cur_org_mid_dir, f"configs")
        os.makedirs(config_dir, exist_ok=True)

        b = min(column_bit_count, 0)
        preset_summary_bits[org_preset] = b
        dram_config_file = os.path.join(config_dir, f"dram_config.yaml")
        dram_config["MemorySystem"]["DRAM"]["org"]["preset"] = org_preset
        dram_config["MemorySystem"]["DRAM"]["org"]["channel"] = channel_num
        dram_config["MemorySystem"]["DRAM"]["org"]["internal_prefetch_size"] = prefetch_size
        dram_config["MemorySystem"]["DRAM"]["timing"]["preset"] = timing_preset
        dram_config["MemorySystem"]["AddrMapper"]["impl"] = "OneLevelInterleave"
        dram_config["MemorySystem"]["AddrMapper"]["channel_lowest_bit"] = b
        with open(dram_config_file, "w") as f:
            yaml.dump(dram_config, f, sort_keys=False, default_flow_style=False)

        chip_config_file = os.path.join(config_dir, f"chip_config.yaml")
        chip_config["architecture"]["dram"]["config_path"] = dram_config_file
        with open(chip_config_file, "w") as f:
            yaml.dump(chip_config, f, sort_keys=False, default_flow_style=False)

        args.mid_dir = os.path.join(cur_org_mid_dir, f"worker_outputs")
        args.output_dir = os.path.join(cur_org_output_dir, f"final_outputs")
        args.config_path = chip_config_file
        os.makedirs(args.mid_dir, exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)

        for test_type in ["model"]:
            args.test_type = test_type

            args.model_name = "opt_6.7b"
            args.model_config_path = "configs/models/opt_6.7b.json"
            args.batch_size = 4
            args.context_length = 2*1024
            args.element_size = 2
            args.buffer_size = 32*1024
            args.system_channel_num = SYSTEM_CHANNEL_NUM
            if not _should_skip_model(args):
                test_full_model_edge(args)

            args.model_name = "palm_8b"
            args.model_config_path = "configs/models/palm_8b.json"
            args.batch_size = 4
            args.context_length = 2*1024
            args.element_size = 2
            args.buffer_size = 32*1024
            args.system_channel_num = SYSTEM_CHANNEL_NUM
            if not _should_skip_model(args):
                test_full_model_edge(args)

            args.model_name = "llama3_8b"
            args.model_config_path = "configs/models/llama3_8b.json"
            args.batch_size = 4
            args.context_length = 4*1024
            args.element_size = 2
            args.buffer_size = 32*1024
            args.system_channel_num = SYSTEM_CHANNEL_NUM
            if not _should_skip_model(args):
                test_full_model_edge(args)

    summary_path = os.path.join(output_root_dir, "summary.csv")
    _write_model_summary_csv(
        summary_path,
        [
            (org_preset, os.path.join(output_root_dir, org_preset),
             [preset_summary_bits[org_preset]])
            for org_preset in org_presets
        ]
    )
    print(f"[DSE] dse_full_space done -> {summary_path.replace('.csv', '_{{latency,bw,bw_util}}.csv')}")


DRAM_EDGE_EXPERIMENT_ALIASES = {
    "interleave": "01_channel_interleaving",
    "01_channel_interleaving": "01_channel_interleaving",
    "io": "02_io_organization",
    "02_io_organization": "02_io_organization",
    "row": "03_logical_row_size",
    "03_logical_row_size": "03_logical_row_size",
    "full": "04_full_space",
    "04_full_space": "04_full_space",
}


def _selected_experiments(args):
    experiment = args.experiment
    if experiment in ("all", "fig20"):
        return ["01_channel_interleaving", "02_io_organization", "03_logical_row_size", "04_full_space"]
    return [DRAM_EDGE_EXPERIMENT_ALIASES[experiment]]


def _run_experiment(exp, args):
    mid_dir = args.mid_dir
    output_dir = args.output_dir
    if exp == "01_channel_interleaving":
        dse_interleave(args)
    elif exp == "02_io_organization":
        dse_io_organization(args)
    elif exp == "03_logical_row_size":
        dse_logical_row_size(args)
    elif exp == "04_full_space":
        dse_full_space(args)
    args.mid_dir = mid_dir
    args.output_dir = output_dir


def _matrix_shape_kind(fname):
    match = re.search(r"_(\d+)x(\d+)x(\d+)\.csv$", fname)
    if not match:
        return None
    shape = tuple(int(x) for x in match.groups())
    if shape == (8, 4096, 32):
        return "fc"
    if shape == (8, 128, 128):
        return "qk"
    if shape == (8, 2048, 8):
        return "sv"
    return None


def _read_edge_matrix_csvs(base_dir, series_name, series_label):
    matrix_dir = os.path.join(base_dir, "matrix")
    rows = []
    if not os.path.isdir(matrix_dir):
        return rows
    staged = {"fc": {}, "qk": {}, "sv": {}}
    for fname in sorted(os.listdir(matrix_dir)):
        if not fname.endswith(".csv"):
            continue
        kind = _matrix_shape_kind(fname)
        if not kind:
            continue
        with open(os.path.join(matrix_dir, fname), "r", newline="") as f:
            for row in csv.DictReader(f):
                if not row or not row.get("tK"):
                    continue
                pair = (int(float(row["tK"])), int(float(row["tN"])))
                order = None
                if kind == "fc" and pair in EDGE_FC_TILE_ORDER:
                    order = EDGE_FC_TILE_ORDER.index(pair)
                elif kind == "qk" and pair in EDGE_QK_TILE_ORDER:
                    order = EDGE_QK_TILE_ORDER.index(pair)
                elif kind == "sv" and pair in EDGE_SV_TILE_ORDER:
                    order = EDGE_SV_TILE_ORDER.index(pair)
                if order is None:
                    continue
                staged[kind][order] = {
                    "tile": pair,
                    "latency_s": to_float(row.get("latency (s)")),
                    "bw_util_pct": to_float(row.get("bw_util (%)")),
                }
    for order, value in sorted(staged["fc"].items()):
        rows.append({
            "kind": "fc",
            "series": series_name,
            "series_label": series_label,
            "x_order": order,
            "x_label": format_tile(value["tile"]),
            "bw_util_pct": value["bw_util_pct"],
        })
    for order in range(len(EDGE_QK_TILE_ORDER)):
        qk = staged["qk"].get(order)
        sv = staged["sv"].get(order)
        if not qk or not sv:
            continue
        total_latency = qk["latency_s"] + sv["latency_s"]
        if total_latency > 0:
            bw_util_pct = (
                qk["bw_util_pct"] * qk["latency_s"] + sv["bw_util_pct"] * sv["latency_s"]
            ) / total_latency
        else:
            bw_util_pct = 0.0
        rows.append({
            "kind": "attention",
            "series": series_name,
            "series_label": series_label,
            "x_order": order,
            "x_label": f"{format_tile(qk['tile'])}\n{format_tile(sv['tile'])}",
            "bw_util_pct": bw_util_pct,
        })
    return rows


def _collect_line_experiment(output_dir, exp):
    root = os.path.join(output_dir, exp)
    rows = []
    if exp == "01_channel_interleaving":
        for bit in [1, 3, 5, 7]:
            rows.extend(_read_edge_matrix_csvs(os.path.join(root, f"interleave_{bit}bit"), f"bit{bit}", f"x = {bit}bit"))
    elif exp == "02_io_organization":
        for channel in [1, 2, 4, 8]:
            rows.extend(_read_edge_matrix_csvs(os.path.join(root, f"channel_num_{channel}", "final_outputs"), f"{channel}ch", f"{channel} Channels"))
    elif exp == "03_logical_row_size":
        for row_size in [2, 4, 8, 16]:
            rows.extend(_read_edge_matrix_csvs(os.path.join(root, f"logical_row_size_{row_size}KB", "final_outputs"), f"{row_size}KB", f"{row_size}KB/LRow"))
    return rows


def _collect_full_space(output_dir):
    root = os.path.join(output_dir, "04_full_space")
    required_models = {"opt_6.7b", "llama3_8b", "palm_8b"}
    rows = []
    if not os.path.isdir(root):
        return rows
    for entry in sorted(os.listdir(root)):
        match = re.match(r"(\d+)channels_(\d+)KB", entry)
        if not match:
            continue
        channel = int(match.group(1))
        row_size = int(match.group(2))
        final_dir = os.path.join(root, entry, "final_outputs")
        if not os.path.isdir(final_dir):
            continue
        for model_dir in sorted(os.listdir(final_dir)):
            csv_path = os.path.join(final_dir, model_dir, "e2e_performance.csv")
            if not os.path.exists(csv_path):
                continue
            model = model_dir.rsplit("_bs", 1)[0]
            with open(csv_path, "r", newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("name") == "total":
                        rows.append({
                            "model": model,
                            "channel_num": channel,
                            "logical_row_size_kb": row_size,
                            "latency_s": to_float(row.get("latency (s)")),
                            "bw_util_pct": to_float(row.get("bw_util (%)")),
                        })
    for channel in [1, 2, 4, 8]:
        for row_size in [2, 4, 8, 16]:
            cell = [r for r in rows if r["channel_num"] == channel and r["logical_row_size_kb"] == row_size]
            models_seen = {r["model"] for r in cell}
            missing_models = sorted(required_models - models_seen)
            if missing_models:
                raise ValueError(
                    f"Missing edge DRAM full-space model results for "
                    f"channel={channel}, logical_row_size_kb={row_size}: {missing_models}"
                )
            total_lat = sum(r["latency_s"] for r in cell)
            if total_lat:
                rows.append({
                    "model": "Average",
                    "channel_num": channel,
                    "logical_row_size_kb": row_size,
                    "latency_s": total_lat,
                    "bw_util_pct": sum(r["bw_util_pct"] * r["latency_s"] for r in cell) / total_lat,
                })
    return rows


def collect_results(args):
    summary_dir = ensure_dir(os.path.join(args.output_dir, "summary"))
    for exp in _selected_experiments(args):
        if exp in ("01_channel_interleaving", "02_io_organization", "03_logical_row_size"):
            rows = _collect_line_experiment(args.output_dir, exp)
            write_csv_dicts(
                os.path.join(summary_dir, f"{exp}_fig20_lines.csv"),
                rows,
                ["kind", "series", "series_label", "x_order", "x_label", "bw_util_pct"],
            )
        elif exp == "04_full_space":
            rows = _collect_full_space(args.output_dir)
            write_csv_dicts(
                os.path.join(summary_dir, "04_full_space_fig20d_heatmap.csv"),
                rows,
                ["model", "channel_num", "logical_row_size_kb", "latency_s", "bw_util_pct"],
            )
    print(f"[COLLECT] edge DRAM summaries -> {summary_dir}")


def _read_summary(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _plot_line_csv(csv_path, output_base, title):
    import matplotlib.pyplot as plt

    rows = _read_summary(csv_path)
    if not rows:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 2.8), sharey=True)
    for ax, kind, panel_title in ((axes[0], "fc", "FC"), (axes[1], "attention", "Attention")):
        panel_rows = [r for r in rows if r["kind"] == kind]
        series_labels = []
        for row in panel_rows:
            if row["series_label"] not in series_labels:
                series_labels.append(row["series_label"])
        for style_idx, label in enumerate(series_labels):
            vals = sorted([r for r in panel_rows if r["series_label"] == label], key=lambda r: int(float(r["x_order"])))
            line_style = dict(EDGE_DRAM_LINE_STYLES[style_idx])
            line_style.update({
                "markeredgecolor": line_style["color"],
                "markeredgewidth": 1.0,
                "markersize": 5.2,
                "linewidth": 1.3,
            })
            ax.plot(
                [r["x_label"] for r in vals],
                [to_float(r["bw_util_pct"]) for r in vals],
                label=label,
                **line_style,
            )
        ax.set_title(panel_title)
        ax.set_ylabel("BW. Util. (%)")
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_xlabel("Tiling")
        ax.tick_params(axis="x", rotation=45, pad=1)
        for label in ax.get_xticklabels():
            label.set_ha("right")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=len(labels), loc="upper center", bbox_to_anchor=(0.5, 0.93))
    fig.suptitle(title, y=0.995)
    fig.subplots_adjust(top=0.78, bottom=0.22, left=0.08, right=0.99, wspace=0.16)
    save_basic_figure(fig, output_base)
    plt.close(fig)
    return True


def _plot_heatmap_csv(csv_path, output_base):
    import matplotlib.pyplot as plt
    import numpy as np

    rows = _read_summary(csv_path)
    if not rows:
        return False
    models = ["opt_6.7b", "llama3_8b", "palm_8b", "Average"]
    titles = ["OPT", "LLaMA", "PaLM", "Average"]
    row_sizes = [2, 4, 8, 16]
    channels = [1, 2, 4, 8]
    fig = plt.figure(figsize=(9.8, 2.7))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.06], wspace=0.16)
    im = None
    for idx, (model, title) in enumerate(zip(models, titles)):
        ax = fig.add_subplot(gs[0, idx])
        data = np.full((len(channels), len(row_sizes)), np.nan)
        for r in rows:
            if r["model"] == model:
                ch = int(float(r["channel_num"]))
                rs = int(float(r["logical_row_size_kb"]))
                if ch in channels and rs in row_sizes:
                    data[channels.index(ch), row_sizes.index(rs)] = to_float(r["bw_util_pct"])
        im = draw_dram_heatmap_pct(
            ax,
            data,
            [str(v) for v in row_sizes],
            [str(v) for v in channels],
            title=title,
            xlabel="Row Size (KB)",
            ylabel="Channel Number" if idx == 0 else "",
            show_yticks=(idx == 0),
            vmin=40,
            vmax=100,
            cell_fontsize=6.2,
            near_max_threshold_pct=0,
        )
    cbar_ax = fig.add_subplot(gs[0, 4])
    add_dram_percent_colorbar(fig, im, cbar_ax, "Bandwidth Utilization", vmin=40, vmax=100)
    save_basic_figure(fig, output_base)
    plt.close(fig)
    return True


def _remove_legacy_figures(figure_dir, *figure_names):
    for figure_name in figure_names:
        for extension in (".png", ".pdf"):
            path = os.path.join(figure_dir, figure_name + extension)
            if os.path.exists(path):
                os.remove(path)


def plot_results(args):
    apply_paper_style()
    summary_dir = os.path.join(args.output_dir, "summary")
    figure_dir = ensure_dir(os.path.join(args.output_dir, "figures"))
    line_figure_names = {
        "01_channel_interleaving": "fig20a_channel_interleaving",
        "02_io_organization": "fig20b_io_organization",
        "03_logical_row_size": "fig20c_logical_row_size",
    }
    for exp in _selected_experiments(args):
        if exp in ("01_channel_interleaving", "02_io_organization", "03_logical_row_size"):
            plotted = _plot_line_csv(
                os.path.join(summary_dir, f"{exp}_fig20_lines.csv"),
                os.path.join(figure_dir, line_figure_names[exp]),
                exp,
            )
            if plotted:
                _remove_legacy_figures(figure_dir, f"{exp}_fig17")
        elif exp == "04_full_space":
            plotted = _plot_heatmap_csv(
                os.path.join(summary_dir, "04_full_space_fig20d_heatmap.csv"),
                os.path.join(figure_dir, "fig20d_full_space_heatmap"),
            )
            if plotted:
                _remove_legacy_figures(figure_dir, "fig17d_full_space_heatmap")


if __name__ == "__main__":
    _start_time = time.perf_counter()
    try:
        args = parse_args()

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.root_dir = os.path.relpath(project_root, os.getcwd())

        args.mid_dir = os.path.join(args.root_dir, args.mid_dir)
        args.output_dir = os.path.join(args.root_dir, args.output_dir)

        os.makedirs(args.mid_dir, exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)

        for exp in _selected_experiments(args):
            if args.action in ("run", "all"):
                _run_experiment(exp, args)
        if args.action in ("collect", "all"):
            collect_results(args)
        if args.action in ("plot", "all"):
            plot_results(args)
    finally:
        print(f"[TIMER] dram_dse_edge.py elapsed: {time.perf_counter() - _start_time:.2f}s")
