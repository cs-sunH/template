import argparse
import csv
import re
import time
import yaml
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dram_test import *
from ae_utils import (
    CLOUD_ATTN_BLOCK_ORDER,
    CLOUD_GEMM_TILE_ORDER,
    add_dram_percent_colorbar,
    apply_paper_style,
    draw_dram_heatmap_pct,
    ensure_dir,
    format_tile,
    save_basic_figure,
    to_float,
    write_csv_dicts,
)


CLOUD_DRAM_LINE_STYLES = [
    {"color": "#2E75B6", "marker": "s", "markerfacecolor": "white"},
    {"color": "#C55A11", "marker": None},
    {"color": "#7C7C7C", "marker": "o", "markerfacecolor": "white"},
    {"color": "#FFC000", "marker": "D", "markerfacecolor": "white"},
    {"color": "#548235", "marker": "^", "markerfacecolor": "white"},
    {"color": "#DE8F8F", "marker": "x", "markerfacecolor": "none"},
]


column_bit_dict = {
    "HBDRAM_4Gb_1024pin" : 7,

    "HBDRAM_2Gb_512pin": 8,
    "HBDRAM_1Gb_256pin": 9,
    "HBDRAM_8Gb_2048pin": 6,
    "HBDRAM_16Gb_4096pin": 5,
    "HBDRAM_32Gb_8192pin": 4,

    "HBDRAM_4Gb_1024pin_256col": 8,
    "HBDRAM_4Gb_1024pin_512col": 9,
    "HBDRAM_4Gb_1024pin_64col": 6,
    "HBDRAM_4Gb_1024pin_32col": 5,
    "HBDRAM_4Gb_1024pin_16col": 4,

    "HBDRAM_2Gb_512pin_32KB": 9,
    "HBDRAM_2Gb_512pin_64KB": 10,
    "HBDRAM_2Gb_512pin_8KB": 7,
    "HBDRAM_2Gb_512pin_4KB": 6,
    "HBDRAM_2Gb_512pin_2KB": 5,

    "HBDRAM_1Gb_256pin_32KB": 10,
    "HBDRAM_1Gb_256pin_64KB": 11,
    "HBDRAM_1Gb_256pin_8KB": 8,
    "HBDRAM_1Gb_256pin_4KB": 7,
    "HBDRAM_1Gb_256pin_2KB": 6,

    "HBDRAM_8Gb_2048pin_32KB": 7,
    "HBDRAM_8Gb_2048pin_64KB": 8,
    "HBDRAM_8Gb_2048pin_8KB": 5,
    "HBDRAM_8Gb_2048pin_4KB": 4,
    "HBDRAM_8Gb_2048pin_2KB": 3,

    "HBDRAM_16Gb_4096pin_32KB": 6,
    "HBDRAM_16Gb_4096pin_64KB": 7,
    "HBDRAM_16Gb_4096pin_8KB": 4,
    "HBDRAM_16Gb_4096pin_4KB": 3,
    "HBDRAM_16Gb_4096pin_2KB": 2,

    "HBDRAM_32Gb_8192pin_32KB": 5,
    "HBDRAM_32Gb_8192pin_64KB": 6,
    "HBDRAM_32Gb_8192pin_8KB": 3,
    "HBDRAM_32Gb_8192pin_4KB": 2,
    "HBDRAM_32Gb_8192pin_2KB": 1,
}


channel_num_dict = {
    "HBDRAM_4Gb_1024pin" : 16,

    "HBDRAM_2Gb_512pin": 32,
    "HBDRAM_1Gb_256pin": 64,
    "HBDRAM_8Gb_2048pin": 8,
    "HBDRAM_16Gb_4096pin": 4,
    "HBDRAM_32Gb_8192pin": 2,

    "HBDRAM_4Gb_1024pin_256col": 16,
    "HBDRAM_4Gb_1024pin_512col": 16,
    "HBDRAM_4Gb_1024pin_64col": 16,
    "HBDRAM_4Gb_1024pin_32col": 16,
    "HBDRAM_4Gb_1024pin_16col": 16,

    "HBDRAM_2Gb_512pin_32KB": 32,
    "HBDRAM_2Gb_512pin_64KB": 32,
    "HBDRAM_2Gb_512pin_8KB": 32,
    "HBDRAM_2Gb_512pin_4KB": 32,
    "HBDRAM_2Gb_512pin_2KB": 32,

    "HBDRAM_1Gb_256pin_32KB": 64,
    "HBDRAM_1Gb_256pin_64KB": 64,
    "HBDRAM_1Gb_256pin_8KB": 64,
    "HBDRAM_1Gb_256pin_4KB": 64,
    "HBDRAM_1Gb_256pin_2KB": 64,

    "HBDRAM_8Gb_2048pin_32KB": 8,
    "HBDRAM_8Gb_2048pin_64KB": 8,
    "HBDRAM_8Gb_2048pin_8KB": 8,
    "HBDRAM_8Gb_2048pin_4KB": 8,
    "HBDRAM_8Gb_2048pin_2KB": 8,

    "HBDRAM_16Gb_4096pin_32KB": 4,
    "HBDRAM_16Gb_4096pin_64KB": 4,
    "HBDRAM_16Gb_4096pin_8KB": 4,
    "HBDRAM_16Gb_4096pin_4KB": 4,
    "HBDRAM_16Gb_4096pin_2KB": 4,

    "HBDRAM_32Gb_8192pin_32KB": 2,
    "HBDRAM_32Gb_8192pin_64KB": 2,
    "HBDRAM_32Gb_8192pin_8KB": 2,
    "HBDRAM_32Gb_8192pin_4KB": 2,
    "HBDRAM_32Gb_8192pin_2KB": 2,
}


logical_row_size_dict = {
    "HBDRAM_4Gb_1024pin" : 16,

    "HBDRAM_2Gb_512pin": 16,
    "HBDRAM_1Gb_256pin": 16,
    "HBDRAM_8Gb_2048pin": 16,
    "HBDRAM_16Gb_4096pin": 16,
    "HBDRAM_32Gb_8192pin": 16,

    "HBDRAM_4Gb_1024pin_256col": 32,
    "HBDRAM_4Gb_1024pin_512col": 64,
    "HBDRAM_4Gb_1024pin_64col": 8,
    "HBDRAM_4Gb_1024pin_32col": 4,
    "HBDRAM_4Gb_1024pin_16col": 2,

    "HBDRAM_2Gb_512pin_32KB": 32,
    "HBDRAM_2Gb_512pin_64KB": 64,
    "HBDRAM_2Gb_512pin_8KB": 8,
    "HBDRAM_2Gb_512pin_4KB": 4,
    "HBDRAM_2Gb_512pin_2KB": 2,

    "HBDRAM_1Gb_256pin_32KB": 32,
    "HBDRAM_1Gb_256pin_64KB": 64,
    "HBDRAM_1Gb_256pin_8KB": 8,
    "HBDRAM_1Gb_256pin_4KB": 4,
    "HBDRAM_1Gb_256pin_2KB": 2,

    "HBDRAM_8Gb_2048pin_32KB": 32,
    "HBDRAM_8Gb_2048pin_64KB": 64,
    "HBDRAM_8Gb_2048pin_8KB": 8,
    "HBDRAM_8Gb_2048pin_4KB": 4,
    "HBDRAM_8Gb_2048pin_2KB": 2,

    "HBDRAM_16Gb_4096pin_32KB": 32,
    "HBDRAM_16Gb_4096pin_64KB": 64,
    "HBDRAM_16Gb_4096pin_8KB": 8,
    "HBDRAM_16Gb_4096pin_4KB": 4,
    "HBDRAM_16Gb_4096pin_2KB": 2,

    "HBDRAM_32Gb_8192pin_32KB": 32,
    "HBDRAM_32Gb_8192pin_64KB": 64,
    "HBDRAM_32Gb_8192pin_8KB": 8,
    "HBDRAM_32Gb_8192pin_4KB": 4,
    "HBDRAM_32Gb_8192pin_2KB": 2,
}


timing_dict = {k: "HBDRAM_500Mbps" for k in column_bit_dict}


prefetch_dict = {k: 1 for k in column_bit_dict}


chip_config = {
    "architecture": {
        "frequency": 1000,
        "core_num": 16,
        "core": {
            "controller": {
                "power": 1.0,
                "area": 1.0,
            },
            "matrix": {
                "mac_num": 8192,
                "power": 1.1,
                "area": 1.1,
            },
            "vector": {
                "vec_num": 512,
                "power": 1.2,
                "area": 1.2,
            },
            "buffer": {
                "buffer_size": 3072,
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
        "noc": {
            "topology": "mesh",
            "config_path": "configs/architecture/noc/4x4_mesh",
            "flit_size": 64,
            "power": 1.5,
            "area": 1.5,
        }
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
                "preset": "HBDRAM_1024Gb_1024pin",
                "channel": 16,
                "internal_prefetch_size": 1,
            },
            "timing": {
                "preset": "HBDRAM_500Mbps"
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="all", choices=["01_channel_interleaving", "02_io_organization", "03_logical_row_size", "04_full_space", "interleave", "io", "row", "full", "fig10", "all"])
    parser.add_argument("--action", type=str, default="all", choices=["run", "collect", "plot", "all"])
    parser.add_argument("--mid-dir", type=str, default="results/dram_dse_cloud/mid_results")
    parser.add_argument("--output-dir", type=str, default="results/dram_dse_cloud/output_results")
    parser.add_argument("--num-workers", type=int, default=int(multiprocessing.cpu_count()*0.8))
    return parser.parse_args()


def dse_interleave(args):
    print("=" * 60)
    print("[DSE] cloud dse_interleave")
    print("=" * 60)
    org_preset = "HBDRAM_4Gb_1024pin"
    timing_preset = timing_dict[org_preset]
    column_bit_count = column_bit_dict[org_preset]
    channel_num = 16

    mid_root_dir = os.path.join(args.mid_dir, f"01_channel_interleaving")
    output_root_dir = os.path.join(args.output_dir, f"01_channel_interleaving")
    os.makedirs(mid_root_dir, exist_ok=True)
    os.makedirs(output_root_dir, exist_ok=True)


    config_dir = os.path.join(mid_root_dir, f"configs")
    os.makedirs(config_dir, exist_ok=True)
    bit_list = [0, 1, 3, 5, 7]
    for idx, b in enumerate(bit_list, 1):
        print(f"[{idx}/{len(bit_list)}] {org_preset}, interleave bit={b}")
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

        for test_type in ["attention", "matrix"]:
            args.test_type = test_type
            if test_type == "matrix":
                args.M = 64
                args.K = 2048
                args.N = 2048
                args.tM_list = [4, 16, 64]
                args.tK_list = [128, 256, 512, 1024, 2048]
                args.tN_list = [128, 512]
                args.test_type = "matrix"
                test_mapping[test_type](args)
            else:
                test_mapping[test_type](args)
    print(f"[DSE] cloud dse_interleave done -> {output_root_dir}")


def dse_io_organization(args):
    print("=" * 60)
    print("[DSE] cloud dse_io_organization")
    print("=" * 60)
    mid_root_dir = os.path.join(args.mid_dir, f"02_io_organization")
    output_root_dir = os.path.join(args.output_dir, f"02_io_organization")
    os.makedirs(mid_root_dir, exist_ok=True)
    os.makedirs(output_root_dir, exist_ok=True)

    org_presets = [
        "HBDRAM_2Gb_512pin",
        "HBDRAM_1Gb_256pin",

        "HBDRAM_8Gb_2048pin",
        "HBDRAM_16Gb_4096pin",
        "HBDRAM_32Gb_8192pin",
        
        "HBDRAM_4Gb_1024pin",
    ]

    for idx, org_preset in enumerate(org_presets, 1):
        timing_preset = timing_dict[org_preset]
        column_bit_count = column_bit_dict[org_preset]
        channel_num = channel_num_dict[org_preset]
        print(f"[{idx}/{len(org_presets)}] {org_preset}, channel_num={channel_num}")

        cur_org_mid_dir = os.path.join(mid_root_dir, f"channel_num_{channel_num}")
        cur_org_output_dir = os.path.join(output_root_dir, f"channel_num_{channel_num}")
        os.makedirs(cur_org_mid_dir, exist_ok=True)
        os.makedirs(cur_org_output_dir, exist_ok=True)

        config_dir = os.path.join(cur_org_mid_dir, f"configs")
        os.makedirs(config_dir, exist_ok=True)

        b = min(column_bit_count, 5)
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

        for test_type in ["attention", "matrix"]:
            args.test_type = test_type
            if test_type == "matrix":
                args.M = 64
                args.K = 2048
                args.N = 2048
                args.tM_list = [4, 16, 64]
                args.tK_list = [128, 256, 512, 1024, 2048]
                args.tN_list = [128, 512]
                args.test_type = "matrix"
                test_mapping[test_type](args)
            else:
                test_mapping[test_type](args)
    print(f"[DSE] cloud dse_io_organization done -> {output_root_dir}")


def dse_logical_row_size(args):
    print("=" * 60)
    print("[DSE] cloud dse_logical_row_size")
    print("=" * 60)
    mid_root_dir = os.path.join(args.mid_dir, f"03_logical_row_size")
    output_root_dir = os.path.join(args.output_dir, f"03_logical_row_size")
    os.makedirs(mid_root_dir, exist_ok=True)
    os.makedirs(output_root_dir, exist_ok=True)

    org_presets = [
        "HBDRAM_4Gb_1024pin_256col",
        "HBDRAM_4Gb_1024pin_512col",
        
        "HBDRAM_4Gb_1024pin_64col",
        "HBDRAM_4Gb_1024pin_32col",
        "HBDRAM_4Gb_1024pin_16col",
        
        "HBDRAM_4Gb_1024pin",
    ]

    for idx, org_preset in enumerate(org_presets, 1):
        timing_preset = timing_dict[org_preset]
        column_bit_count = column_bit_dict[org_preset]
        channel_num = channel_num_dict[org_preset]
        logical_row_size = logical_row_size_dict[org_preset]
        print(f"[{idx}/{len(org_presets)}] {org_preset}, logical_row_size={logical_row_size}KB")

        cur_org_mid_dir = os.path.join(mid_root_dir, f"logical_row_size_{logical_row_size}KB")
        cur_org_output_dir = os.path.join(output_root_dir, f"logical_row_size_{logical_row_size}KB")
        os.makedirs(cur_org_mid_dir, exist_ok=True)
        os.makedirs(cur_org_output_dir, exist_ok=True)

        config_dir = os.path.join(cur_org_mid_dir, f"configs")
        os.makedirs(config_dir, exist_ok=True)

        b = min(column_bit_count, 5)
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

        for test_type in ["attention", "matrix"]:
            args.test_type = test_type
            if test_type == "matrix":
                args.M = 64
                args.K = 2048
                args.N = 2048
                args.tM_list = [4, 16, 64]
                args.tK_list = [128, 256, 512, 1024, 2048]
                args.tN_list = [128, 512]
                args.test_type = "matrix"
                test_mapping[test_type](args)
            else:
                test_mapping[test_type](args)
    print(f"[DSE] cloud dse_logical_row_size done -> {output_root_dir}")


def dse_full_space(args):
    print("=" * 60)
    print("[DSE] cloud dse_full_space")
    print("=" * 60)
    mid_root_dir = os.path.join(args.mid_dir, f"04_full_space")
    output_root_dir = os.path.join(args.output_dir, f"04_full_space")
    os.makedirs(mid_root_dir, exist_ok=True)
    os.makedirs(output_root_dir, exist_ok=True)

    org_presets = [
        "HBDRAM_4Gb_1024pin",
        "HBDRAM_4Gb_1024pin_256col",
        "HBDRAM_4Gb_1024pin_512col",
        "HBDRAM_4Gb_1024pin_64col",
        "HBDRAM_4Gb_1024pin_32col",
        "HBDRAM_4Gb_1024pin_16col",

        "HBDRAM_2Gb_512pin",
        "HBDRAM_2Gb_512pin_32KB",
        "HBDRAM_2Gb_512pin_64KB",
        "HBDRAM_2Gb_512pin_8KB",
        "HBDRAM_2Gb_512pin_4KB",
        "HBDRAM_2Gb_512pin_2KB",

        "HBDRAM_1Gb_256pin",
        "HBDRAM_1Gb_256pin_32KB",
        "HBDRAM_1Gb_256pin_64KB",
        "HBDRAM_1Gb_256pin_8KB",
        "HBDRAM_1Gb_256pin_4KB",
        "HBDRAM_1Gb_256pin_2KB",

        "HBDRAM_8Gb_2048pin",
        "HBDRAM_8Gb_2048pin_32KB",
        "HBDRAM_8Gb_2048pin_64KB",
        "HBDRAM_8Gb_2048pin_8KB",
        "HBDRAM_8Gb_2048pin_4KB",
        "HBDRAM_8Gb_2048pin_2KB",

        "HBDRAM_16Gb_4096pin",
        "HBDRAM_16Gb_4096pin_32KB",
        "HBDRAM_16Gb_4096pin_64KB",
        "HBDRAM_16Gb_4096pin_8KB",
        "HBDRAM_16Gb_4096pin_4KB",
        "HBDRAM_16Gb_4096pin_2KB",

        "HBDRAM_32Gb_8192pin",
        "HBDRAM_32Gb_8192pin_32KB",
        "HBDRAM_32Gb_8192pin_64KB",
        "HBDRAM_32Gb_8192pin_8KB",
        "HBDRAM_32Gb_8192pin_4KB",
        "HBDRAM_32Gb_8192pin_2KB",
    ]

    for idx, org_preset in enumerate(org_presets, 1):
        timing_preset = timing_dict[org_preset]
        column_bit_count = column_bit_dict[org_preset]
        channel_num = channel_num_dict[org_preset]
        prefetch_size = prefetch_dict[org_preset]
        logical_row_size = logical_row_size_dict[org_preset]
        print(
            f"[{idx}/{len(org_presets)}] {org_preset}, "
            f"channel_num={channel_num}, logical_row_size={logical_row_size}KB"
        )

        cur_org_mid_dir = os.path.join(mid_root_dir, f"{channel_num}channels_{logical_row_size}KB")
        cur_org_output_dir = os.path.join(output_root_dir, f"{channel_num}channels_{logical_row_size}KB")
        os.makedirs(cur_org_mid_dir, exist_ok=True)
        os.makedirs(cur_org_output_dir, exist_ok=True)

        config_dir = os.path.join(cur_org_mid_dir, f"configs")
        os.makedirs(config_dir, exist_ok=True)

        b = min(column_bit_count, 5)
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

            args.model_name = "opt_66b"
            args.model_config_path = "configs/models/opt_66b.json"
            args.batch_size = 64
            args.context_length = 4*1024
            args.element_size = 2
            args.block_size = 128
            args.buffer_size = 1*1024*1024
            test_mapping[test_type](args)

            args.model_name = "llama3_70b"
            args.model_config_path = "configs/models/llama3_70b.json"
            args.batch_size = 64
            args.context_length = 16*1024
            args.element_size = 2
            args.block_size = 128
            args.buffer_size = 1*1024*1024
            test_mapping[test_type](args)

            args.model_name = "mixtral_8x22b"
            args.model_config_path = "configs/models/mixtral_8x22b.json"
            args.batch_size = 64
            args.context_length = 16*1024
            args.element_size = 2
            args.block_size = 128
            args.buffer_size = 1*1024*1024
            test_mapping[test_type](args)

            args.model_name = "qwen3_235b_a22b"
            args.model_config_path = "configs/models/qwen3_235b_a22b.json"
            args.batch_size = 64
            args.context_length = 4*1024
            args.element_size = 1
            args.block_size = 128
            args.buffer_size = 1*1024*1024
            test_mapping[test_type](args)
    print(f"[DSE] cloud dse_full_space done -> {output_root_dir}")


DRAM_CLOUD_EXPERIMENT_ALIASES = {
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
    if experiment in ("all", "fig10"):
        return ["01_channel_interleaving", "02_io_organization", "03_logical_row_size", "04_full_space"]
    return [DRAM_CLOUD_EXPERIMENT_ALIASES[experiment]]


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


def _read_matrix_csvs(base_dir, series_name, series_label):
    rows = []
    matrix_dir = os.path.join(base_dir, "matrix")
    if not os.path.isdir(matrix_dir):
        return rows
    for fname in sorted(os.listdir(matrix_dir)):
        if not fname.endswith(".csv"):
            continue
        with open(os.path.join(matrix_dir, fname), "r", newline="") as f:
            for row in csv.DictReader(f):
                if not row or not row.get("tM"):
                    continue
                tK = int(float(row["tK"]))
                tN = int(float(row["tN"]))
                if (tK, tN) not in CLOUD_GEMM_TILE_ORDER:
                    continue
                rows.append({
                    "series": series_name,
                    "series_label": series_label,
                    "panel": f"tM = {int(float(row['tM']))}",
                    "tM": int(float(row["tM"])),
                    "tK": tK,
                    "tN": tN,
                    "x_order": CLOUD_GEMM_TILE_ORDER.index((tK, tN)),
                    "x_label": format_tile((tK, tN)),
                    "bw_util_pct": to_float(row.get("bw_util (%)")),
                })
    return rows


def _read_attention_csvs(base_dir, series_name, series_label):
    rows = []
    attention_dir = os.path.join(base_dir, "attention")
    if not os.path.isdir(attention_dir):
        return rows
    for fname in sorted(os.listdir(attention_dir)):
        if not fname.endswith(".csv"):
            continue
        with open(os.path.join(attention_dir, fname), "r", newline="") as f:
            for row in csv.DictReader(f):
                block = int(float(row["block_size"]))
                if block not in CLOUD_ATTN_BLOCK_ORDER:
                    continue
                context_tokens = int(float(row["block_size"])) * int(float(row["access_block_num"])) * 16
                if context_tokens not in (16 * 1024, 32 * 1024, 64 * 1024):
                    continue
                rows.append({
                    "series": series_name,
                    "series_label": series_label,
                    "panel": f"{context_tokens // 1024}K Tokens",
                    "context_tokens": context_tokens,
                    "block_size": block,
                    "x_order": CLOUD_ATTN_BLOCK_ORDER.index(block),
                    "x_label": str(block),
                    "bw_util_pct": to_float(row.get("bw_util (%)")),
                })
    return rows


def _collect_line_experiment(output_dir, exp):
    root = os.path.join(output_dir, exp)
    rows = []
    if exp == "01_channel_interleaving":
        for bit in [0, 1, 3, 5, 7]:
            base = os.path.join(root, f"interleave_{bit}bit")
            rows.extend({**r, "kind": "gemm"} for r in _read_matrix_csvs(base, f"bit{bit}", f"x = {bit}bit"))
            rows.extend({**r, "kind": "attention"} for r in _read_attention_csvs(base, f"bit{bit}", f"x = {bit}bit"))
    elif exp == "02_io_organization":
        for channel in [2, 4, 8, 16, 32, 64]:
            base = os.path.join(root, f"channel_num_{channel}", "final_outputs")
            rows.extend({**r, "kind": "gemm"} for r in _read_matrix_csvs(base, f"{channel}ch", f"{channel} Channels"))
            rows.extend({**r, "kind": "attention"} for r in _read_attention_csvs(base, f"{channel}ch", f"{channel} Channels"))
    elif exp == "03_logical_row_size":
        for row_size in [2, 4, 8, 16, 32, 64]:
            base = os.path.join(root, f"logical_row_size_{row_size}KB", "final_outputs")
            rows.extend({**r, "kind": "gemm"} for r in _read_matrix_csvs(base, f"{row_size}KB", f"{row_size}KB/LRow"))
            rows.extend({**r, "kind": "attention"} for r in _read_attention_csvs(base, f"{row_size}KB", f"{row_size}KB/LRow"))
    return rows


def _collect_full_space(output_dir):
    root = os.path.join(output_dir, "04_full_space")
    required_models = {"opt_66b", "llama3_70b", "mixtral_8x22b", "qwen3_235b_a22b"}
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
    for channel in [2, 4, 8, 16, 32, 64]:
        for row_size in [2, 4, 8, 16, 32, 64]:
            cell = [r for r in rows if r["channel_num"] == channel and r["logical_row_size_kb"] == row_size]
            models_seen = {r["model"] for r in cell}
            missing_models = sorted(required_models - models_seen)
            if missing_models:
                raise ValueError(
                    f"Missing cloud DRAM full-space model results for "
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
                os.path.join(summary_dir, f"{exp}_fig10_lines.csv"),
                rows,
                ["kind", "series", "series_label", "panel", "tM", "tK", "tN", "context_tokens", "block_size", "x_order", "x_label", "bw_util_pct"],
            )
        elif exp == "04_full_space":
            rows = _collect_full_space(args.output_dir)
            write_csv_dicts(
                os.path.join(summary_dir, "04_full_space_fig10d_heatmap.csv"),
                rows,
                ["model", "channel_num", "logical_row_size_kb", "latency_s", "bw_util_pct"],
            )
    print(f"[COLLECT] cloud DRAM summaries -> {summary_dir}")


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
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 5.8), sharey=True)
    panels = {
        "gemm": ["tM = 4", "tM = 16", "tM = 64"],
        "attention": ["16K Tokens", "32K Tokens", "64K Tokens"],
    }
    for row_idx, kind in enumerate(("gemm", "attention")):
        for col_idx, panel in enumerate(panels[kind]):
            ax = axes[row_idx][col_idx]
            panel_rows = [r for r in rows if r["kind"] == kind and r["panel"] == panel]
            series_labels = []
            for row in panel_rows:
                if row["series_label"] not in series_labels:
                    series_labels.append(row["series_label"])
            for style_idx, label in enumerate(series_labels):
                vals = sorted([r for r in panel_rows if r["series_label"] == label], key=lambda r: int(float(r["x_order"])))
                line_style = dict(CLOUD_DRAM_LINE_STYLES[style_idx])
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
            ax.set_title(panel)
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            ax.tick_params(axis="x", rotation=45, pad=1)
            for label in ax.get_xticklabels():
                label.set_ha("right")
            ax.set_ylabel("GEMM BW. Util. (%)" if kind == "gemm" else "Attn. BW. Util. (%)")
            ax.set_xlabel("(tK, tN)" if kind == "gemm" else "Block Size")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=len(labels), loc="upper center", bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(title, y=0.995)
    fig.subplots_adjust(top=0.83, bottom=0.15, left=0.07, right=0.99, hspace=0.82, wspace=0.28)
    save_basic_figure(fig, output_base)
    plt.close(fig)
    return True


def _plot_full_space_heatmaps(csv_path, model_output_base, average_output_base):
    import matplotlib.pyplot as plt
    import numpy as np

    rows = _read_summary(csv_path)
    if not rows:
        return False
    models = ["opt_66b", "llama3_70b", "mixtral_8x22b", "qwen3_235b_a22b"]
    titles = ["OPT", "LLaMA", "Mixtral", "Qwen"]
    row_sizes = [2, 4, 8, 16, 32, 64]
    channels = [2, 4, 8, 16, 32, 64]
    data_by_model = {}
    for model in [*models, "Average"]:
        data = np.full((len(channels), len(row_sizes)), np.nan)
        for r in rows:
            if r["model"] == model:
                ch = int(float(r["channel_num"]))
                rs = int(float(r["logical_row_size_kb"]))
                if ch in channels and rs in row_sizes:
                    data[channels.index(ch), row_sizes.index(rs)] = to_float(r["bw_util_pct"])
        data_by_model[model] = data

    fig = plt.figure(figsize=(11.2, 2.9))
    gs = fig.add_gridspec(
        1,
        5,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 0.08],
        wspace=0.16,
    )
    im = None
    for col_idx, (model, title) in enumerate(zip(models, titles)):
        ax = fig.add_subplot(gs[0, col_idx])
        im = draw_dram_heatmap_pct(
            ax,
            data_by_model[model],
            [str(v) for v in row_sizes],
            [str(v) for v in channels],
            title=title,
            xlabel="Row Size (KB)",
            ylabel="Channel Number" if col_idx == 0 else "",
            show_yticks=col_idx == 0,
            vmin=0,
            vmax=100,
            cell_fontsize=5.5,
            near_max_threshold_pct=1,
        )
    cbar_ax = fig.add_subplot(gs[0, 4])
    add_dram_percent_colorbar(fig, im, cbar_ax, "Bandwidth Utilization", vmin=0, vmax=100)
    save_basic_figure(fig, model_output_base)
    plt.close(fig)

    fig = plt.figure(figsize=(3.4, 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.08], wspace=0.22)
    ax_avg = fig.add_subplot(gs[0, 0])
    im = draw_dram_heatmap_pct(
        ax_avg,
        data_by_model["Average"],
        [str(v) for v in row_sizes],
        [str(v) for v in channels],
        title="Average",
        xlabel="Row Size (KB)",
        ylabel="Channel Number",
        vmin=0,
        vmax=100,
        cell_fontsize=7,
        near_max_threshold_pct=1,
    )
    cbar_ax = fig.add_subplot(gs[0, 1])
    add_dram_percent_colorbar(fig, im, cbar_ax, "Bandwidth Utilization", vmin=0, vmax=100)
    save_basic_figure(fig, average_output_base)
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
        "01_channel_interleaving": "fig10a_channel_interleaving",
        "02_io_organization": "fig10b_io_organization",
        "03_logical_row_size": "fig10c_logical_row_size",
    }
    for exp in _selected_experiments(args):
        if exp in ("01_channel_interleaving", "02_io_organization", "03_logical_row_size"):
            plotted = _plot_line_csv(
                os.path.join(summary_dir, f"{exp}_fig10_lines.csv"),
                os.path.join(figure_dir, line_figure_names[exp]),
                exp,
            )
            if plotted:
                _remove_legacy_figures(
                    figure_dir,
                    f"{exp}_fig11",
                    line_figure_names[exp].replace("fig10", "fig11", 1),
                )
        elif exp == "04_full_space":
            plotted = _plot_full_space_heatmaps(
                os.path.join(summary_dir, "04_full_space_fig10d_heatmap.csv"),
                os.path.join(figure_dir, "fig10d_full_space_models"),
                os.path.join(figure_dir, "fig12_full_space_average"),
            )
            if plotted:
                _remove_legacy_figures(
                    figure_dir,
                    "fig11d_full_space_heatmap",
                    "fig11d_full_space_models",
                )


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
        print(f"[TIMER] dram_dse_cloud.py elapsed: {time.perf_counter() - _start_time:.2f}s")
