import argparse
import copy
import csv
import multiprocessing
import os
import pickle
import re
import sys
import time

import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from frontend.hardware_parser import CloudSystemConfig
from frontend.model_parser import get_model_config_from_hf
from frontend.util import set_pdeathsig
from inference_test_auto import test_cloud_inference
from ae_utils import (
    CHIP_DSE_BASELINE_COLORS,
    CHIP_DSE_COLORS,
    CHIP_DSE_CONFIG_COLORS,
    CLOUD_CASE_ORDER,
    CLOUD_DENSE_MODELS,
    CLOUD_MODEL_LABELS,
    apply_paper_style,
    chip_component_areas,
    chip_component_powers,
    cloud_case_label,
    cloud_dram_bw_tbps,
    copy_file,
    ensure_dir,
    logic_and_dram_power,
    load_yaml,
    matrix_compute_tflops,
    nested_get,
    read_architecture_summary,
    read_temperature_summary,
    save_basic_figure,
    set_paper_y_axis,
    sort_cloud_cases,
    to_float,
    validate_chip_area,
    vector_compute_tflops,
    write_csv_dicts,
)


core_size = (7.0e-3, 7.0e-3)
core_array_size = (4, 4)
base_frequency = 1000
temperature_threshold = 85
num_dram_layer = 4
cooling_style = "liquid"
interposer_scaling_factor = 1.05

system_dict = {
    "system": {
        "type": "cloud",
        "chip": {"config_path": None},
        "parallelism": {
            "tp_size": 8,
            "ep_size": 8,
            "tp_scope": "scale_up",
            "ep_scope": "scale_up",
        },
        "interconnect": {
            "scale_up_latency": 4e-7,
            "scale_up_bandwidth": 900,
            "scale_up_energy": 1.3,
            "scale_out_latency": 2e-6,
            "scale_out_bandwidth": 400,
            "scale_out_energy": 1.3,
            "pcie_bandwidth": 512,
        },
        "kv_cache": {
            "max_context_length": 2097152,
            "block_size": 1024,
        },
    }
}


H200_BASELINE = {
    ("opt_66b", 16, 1024): {"latency_s": 0.0063803, "energy_J": 4.4662},
    ("opt_66b", 16, 4096): {"latency_s": 0.0106944, "energy_J": 7.4861},
    ("opt_66b", 64, 1024): {"latency_s": 0.0109610, "energy_J": 7.6727},
    ("opt_66b", 64, 4096): {"latency_s": 0.0282176, "energy_J": 19.7523},
    ("llama3_70b", 16, 8192): {"latency_s": 0.0067892, "energy_J": 4.7525},
    ("llama3_70b", 16, 32768): {"latency_s": 0.0115827, "energy_J": 8.1079},
    ("llama3_70b", 64, 8192): {"latency_s": 0.0118777, "energy_J": 8.3144},
    ("llama3_70b", 64, 32768): {"latency_s": 0.0310517, "energy_J": 21.7362},
    ("mixtral_8x22b", 16, 8192): {"latency_s": 0.0082356, "energy_J": 5.7649},
    ("mixtral_8x22b", 16, 32768): {"latency_s": 0.0115910, "energy_J": 8.1137},
    ("mixtral_8x22b", 64, 8192): {"latency_s": 0.0117087, "energy_J": 8.1961},
    ("mixtral_8x22b", 64, 32768): {"latency_s": 0.0251304, "energy_J": 17.5913},
    ("qwen3_235b_a22b", 16, 1024): {"latency_s": 0.0177431, "energy_J": 12.4202},
    ("qwen3_235b_a22b", 16, 4096): {"latency_s": 0.0184471, "energy_J": 12.9130},
    ("qwen3_235b_a22b", 64, 1024): {"latency_s": 0.0186971, "energy_J": 13.0880},
    ("qwen3_235b_a22b", 64, 4096): {"latency_s": 0.0215133, "energy_J": 15.0593},
}

CLOUD_MODEL_CONFIG_LIST = [
    ("opt_66b", "configs/models/opt_66b.json", 2, [16, 64], [1024, 4096]),
    ("llama3_70b", "configs/models/llama3_70b.json", 2, [16, 64], [8192, 32768]),
    ("mixtral_8x22b", "configs/models/mixtral_8x22b.json", 2, [16, 64], [8192, 32768]),
    ("qwen3_235b_a22b", "configs/models/qwen3_235b_a22b.json", 2, [16, 64], [1024, 4096]),
]

CLOUD_CHIP_EXPERIMENTS = {
    "01_bandwidth_allocation": {
        "aliases": ("bandwidth", "fig13", "13"),
        "config_dir": "configs/architecture/chip/cloud/comp_bw", "figure_id": "fig13a",
    },
    "02_sram_allocation": {
        "aliases": ("sram", "fig14", "14"),
        "config_dir": "configs/architecture/chip/cloud/sram", "figure_id": "fig14a",
    },
    "03_matrix_vector_allocation": {
        "aliases": ("matrix_vector", "mv", "fig15", "15"),
        "config_dir": "configs/architecture/chip/cloud/matrix_vector", "figure_id": "fig15a",
    },
    "04_noc_allocation": {
        "aliases": ("noc", "fig16", "16"),
        "config_dir": "configs/architecture/chip/cloud/noc", "figure_id": "fig16a",
    },
    "05_baseline_comparison": {
        "aliases": ("baseline", "fig17", "17", "fig18", "18", "fig19", "19"),
        "config_dir": "configs/architecture/chip/cloud/stratum",
    },
}


def get_h200_baseline(model_name, batch_size, context_length):
    return H200_BASELINE.get((model_name, int(batch_size), int(context_length)))


def classify_intra_chip_operator(op_name):
    lower = op_name.lower()
    if lower.endswith("_qk") or lower.endswith("_sv"):
        return "gemm"
    if "attn" in lower or "attention" in lower:
        return "attention"
    if "comm" in lower:
        return "communication"
    return "gemm"


def _fmt_row(row):
    return [f"{v:.17g}" if isinstance(v, float) else v for v in row]


def write_test_case_csv(output_dir, intra_perf, inter_perf, model_config):
    frequency_hz = intra_perf.chip_frequency * 1e6
    ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, "performance.csv")
    category_latency = {"gemm": 0.0, "attention": 0.0, "communication": 0.0}
    category_energy = {"gemm": 0.0, "attention": 0.0, "communication": 0.0}

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "operator_name", "e2e_latency(s)", "e2e_cycles", "e2e_energy(J)",
            "matrix_cycles", "vector_cycles", "buffer_cycles", "dram_cycles", "noc_cycles",
            "matrix_energy(J)", "vector_energy(J)", "buffer_energy(J)", "dram_energy(J)", "noc_energy(J)",
            "matrix_util", "vector_util", "buffer_util", "dram_util", "noc_util",
        ])
        for op_name, op_stats in intra_perf.operator_stats:
            lat = op_stats.e2e_cycles / frequency_hz
            eng = op_stats.e2e_energy
            cat = classify_intra_chip_operator(op_name)
            writer.writerow(_fmt_row([
                op_name, lat, op_stats.e2e_cycles, eng,
                op_stats.matrix_cycles, op_stats.vector_cycles, op_stats.buffer_cycles,
                op_stats.dram_cycles, op_stats.noc_cycles,
                op_stats.matrix_energy, op_stats.vector_energy, op_stats.buffer_energy,
                op_stats.dram_energy, op_stats.noc_energy,
                op_stats.matrix_util(), op_stats.vector_util(), op_stats.buffer_util(),
                op_stats.dram_util(), op_stats.noc_util(),
            ]))
            category_latency[cat] += lat
            category_energy[cat] += eng

        writer.writerow([])
        writer.writerow(["category", "total_latency(s)", "total_energy(J)"])
        for cat in ("gemm", "attention", "communication"):
            writer.writerow(_fmt_row([cat, category_latency[cat], category_energy[cat]]))

        writer.writerow([])
        writer.writerow(["inter_chip_operator_name", "e2e_latency(s)", "e2e_energy(J)"])
        inter_total_lat = 0.0
        inter_total_eng = 0.0
        for op_name, (lat, eng) in inter_perf.items():
            writer.writerow(_fmt_row([op_name, lat, eng]))
            inter_total_lat += lat
            inter_total_eng += eng

        intra_total_lat = intra_perf.e2e_stats.e2e_cycles / frequency_hz
        intra_total_eng = intra_perf.e2e_stats.e2e_energy
        writer.writerow([])
        writer.writerow(["", "total_latency(s)", "total_energy(J)"])
        writer.writerow(_fmt_row(["intra_chip_total", intra_total_lat, intra_total_eng]))
        writer.writerow(_fmt_row(["inter_chip_total", inter_total_lat, inter_total_eng]))

        single_layer_lat = intra_total_lat + inter_total_lat
        single_layer_eng = intra_total_eng + inter_total_eng
        writer.writerow(_fmt_row(["single_layer_total", single_layer_lat, single_layer_eng]))
        all_layers_lat = single_layer_lat * model_config.num_layers
        all_layers_eng = single_layer_eng * model_config.num_layers
        writer.writerow(_fmt_row(["all_layers_total", all_layers_lat, all_layers_eng]))

    return {
        "all_layers_latency": all_layers_lat,
        "all_layers_energy": all_layers_eng,
        "gemm_latency": category_latency["gemm"],
        "attention_latency": category_latency["attention"],
        "intra_comm_latency": category_latency["communication"],
        "inter_comm_latency": inter_total_lat,
        "gemm_energy": category_energy["gemm"],
        "attention_energy": category_energy["attention"],
        "intra_comm_energy": category_energy["communication"],
        "inter_comm_energy": inter_total_eng,
    }


def write_architecture_summary_csv(output_dir, arch_results):
    ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, "architecture_summary.csv")
    metric_keys = [
        "all_layers_latency", "all_layers_energy",
        "gemm_latency", "attention_latency", "intra_comm_latency", "inter_comm_latency",
        "gemm_energy", "attention_energy", "intra_comm_energy", "inter_comm_energy",
    ]
    headers = [
        "model", "batch_size", "context_length",
        "all_layers_latency(s)", "all_layers_energy(J)",
        "single_layer_gemm_latency(s)", "single_layer_attention_latency(s)",
        "single_layer_intra_comm_latency(s)", "single_layer_inter_comm_latency(s)",
        "single_layer_gemm_energy(J)", "single_layer_attention_energy(J)",
        "single_layer_intra_comm_energy(J)", "single_layer_inter_comm_energy(J)",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for (model_name, bs, ctx), result in arch_results:
            writer.writerow(_fmt_row([model_name, bs, ctx] + [result[k] for k in metric_keys]))


def _resolve_cloud_experiment(experiment):
    if experiment == "all":
        return "all"
    for key, meta in CLOUD_CHIP_EXPERIMENTS.items():
        if experiment == key or experiment in meta["aliases"]:
            return key
    raise ValueError(f"Unknown cloud chip experiment: {experiment}")


def _cloud_experiment_list(experiment):
    resolved = _resolve_cloud_experiment(experiment)
    return list(CLOUD_CHIP_EXPERIMENTS.keys()) if resolved == "all" else [resolved]


def _first_number(text, default=0):
    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else default


def _cloud_config_record(experiment, path):
    name = os.path.splitext(os.path.basename(path))[0]
    if experiment == "01_bandwidth_allocation":
        value = _first_number(name)
        return {"name": name, "path": path, "order": value, "label": str(value), "channel_num": value}
    if experiment == "02_sram_allocation":
        value = _first_number(name)
        return {"name": name, "path": path, "order": value, "label": f"{value}MB", "sram_mb": value}
    if experiment == "03_matrix_vector_allocation":
        match = re.search(r"matrix(\d+)_vector(\d+)", name)
        matrix = int(match.group(1)) if match else _first_number(name)
        vector = int(match.group(2)) if match else 1
        return {"name": name, "path": path, "order": matrix, "label": f"{matrix}:{vector}", "matrix_ratio": matrix}
    if experiment == "04_noc_allocation":
        value = _first_number(name)
        return {"name": name, "path": path, "order": value, "label": f"{value}B", "noc_flit_B": value}
    label = "Stratum" if name == "stratum" else "ATLAS"
    order = 0 if name == "stratum" else 1
    return {"name": name, "path": path, "order": order, "label": label}


def _cloud_config_records(experiment):
    config_dir = CLOUD_CHIP_EXPERIMENTS[experiment]["config_dir"]
    records = []
    for fname in os.listdir(config_dir):
        if fname.endswith((".yaml", ".yml")):
            records.append(_cloud_config_record(experiment, os.path.join(config_dir, fname)))
    return sorted(records, key=lambda r: (r["order"], r["name"]))


def _cloud_chip_dir(output_dir, experiment, config_name):
    return os.path.join(output_dir, experiment, config_name)


def _write_cloud_system_config(chip_path, chip_output_dir):
    system_config = copy.deepcopy(system_dict)
    system_config["system"]["chip"]["config_path"] = chip_path
    system_path = os.path.join(chip_output_dir, "system.yaml")
    with open(system_path, "w") as f:
        yaml.dump(system_config, f, sort_keys=False, default_flow_style=False)
    return system_path


def _run_cloud_thermal(config_name, chip_yaml, chip_output_dir):
    from pyta.evaluator import thermal_evaluator

    logic_power, dram_power = logic_and_dram_power(chip_yaml)
    thermal_dir = ensure_dir(os.path.join(chip_output_dir, "thermal"))
    print(f"  [THERMAL] {config_name}: logic_power={logic_power:.4g}W, dram_power={dram_power:.4g}W", flush=True)
    result = thermal_evaluator(
        name=config_name,
        tile_array_size=core_array_size,
        tile_shape=core_size,
        num_memory_layer=num_dram_layer,
        cooling_style=cooling_style,
        interposer_size=max(core_size[i] * core_array_size[i] for i in range(len(core_size))) * interposer_scaling_factor,
        base_frequency=base_frequency,
        logic_power=logic_power,
        dram_power=dram_power,
        temperature_threshold=temperature_threshold,
        frequency_power_dict=None,
        output_dir=thermal_dir,
    ) or {}
    write_csv_dicts(
        os.path.join(thermal_dir, "thermal_summary.csv"),
        [{
            "config_name": config_name,
            "max_temperature_c": result.get("max_temperature_c", ""),
            "frequency_MHz": result.get("frequency_MHz", ""),
            "logic_power_W": logic_power,
            "dram_power_W": dram_power,
        }],
        ["config_name", "max_temperature_c", "frequency_MHz", "logic_power_W", "dram_power_W"],
    )


def _run_cloud_chip_config(task):
    experiment = task["experiment"]
    record = task["record"]
    config_idx = task["config_idx"]
    config_total = task["config_total"]
    chip_output_dir = _cloud_chip_dir(task["output_dir"], experiment, record["name"])
    chip_intermediate_dir = os.path.join(task["mid_dir"], experiment, record["name"])
    print(
        f"[{config_idx}/{config_total}] {experiment}/{record['name']} "
        f"({record['label']})",
        flush=True,
    )
    ensure_dir(chip_output_dir)
    ensure_dir(chip_intermediate_dir)

    chip_path = os.path.join(chip_output_dir, "chip.yaml")
    copy_file(record["path"], chip_path)
    chip_yaml = load_yaml(chip_path)
    validate_chip_area(chip_yaml, core_size)
    system_path = _write_cloud_system_config(chip_path, chip_output_dir)

    thermal_summary = os.path.join(chip_output_dir, "thermal", "thermal_summary.csv")
    if task["run_thermal"] and (task["no_cache"] or not os.path.exists(thermal_summary)):
        _run_cloud_thermal(record["name"], chip_yaml, chip_output_dir)

    with open(system_path, "r") as f:
        cloud_config = CloudSystemConfig.from_yaml(yaml.load(f, Loader=yaml.FullLoader))

    arch_results = []
    total_cases = sum(len(batch_sizes) * len(context_lengths) for _, _, _, batch_sizes, context_lengths in CLOUD_MODEL_CONFIG_LIST)
    case_idx = 0
    for model_name, model_config_path, element_size, batch_sizes, context_lengths in CLOUD_MODEL_CONFIG_LIST:
        model_config = get_model_config_from_hf(model_name, model_config_path)
        for batch_size in batch_sizes:
            gemm_cache_dir = ensure_dir(os.path.join(chip_intermediate_dir, f"{model_name}_bs{batch_size}", "gemm_tiling_cache"))
            for context_length in context_lengths:
                case_idx += 1
                case_name = f"{model_name}_bs{batch_size}_ctx{context_length}"
                case_mid_dir = ensure_dir(os.path.join(chip_intermediate_dir, case_name))
                case_out_dir = ensure_dir(os.path.join(chip_output_dir, case_name))
                pkl_path = os.path.join(case_out_dir, "raw_performance.pkl")
                if not task["no_cache"] and os.path.exists(pkl_path):
                    try:
                        with open(pkl_path, "rb") as f:
                            cached = pickle.load(f)
                        if "case_result" in cached:
                            arch_results.append(((model_name, batch_size, context_length), cached["case_result"]))
                            print(f"  [{case_idx}/{total_cases}] [CACHED] {case_name}", flush=True)
                            continue
                    except Exception as exc:
                        print(f"[WARN] Failed to read cache {pkl_path}: {exc}")

                print(f"  [{case_idx}/{total_cases}] [RUN] {case_name}", flush=True)
                inference_result = test_cloud_inference(
                    cloud_config=cloud_config,
                    model_config=model_config,
                    context_length_list=[context_length] * batch_size,
                    element_size=element_size,
                    intermediate_result_dir=case_mid_dir,
                    gemm_tiling_cache_dir=gemm_cache_dir,
                )
                case_result = write_test_case_csv(
                    case_out_dir,
                    inference_result["intra_chip_computation_performance"],
                    inference_result["inter_chip_communication_performance"],
                    model_config,
                )
                arch_results.append(((model_name, batch_size, context_length), case_result))
                with open(pkl_path, "wb") as f:
                    pickle.dump({"case_result": case_result}, f)
                print(f"  [{case_idx}/{total_cases}] [DONE] {case_name}", flush=True)

    write_architecture_summary_csv(chip_output_dir, arch_results)
    print(f"[DONE] {experiment}/{record['name']} -> {chip_output_dir}", flush=True)
    return {"config": f"{experiment}/{record['name']}", "num_cases": len(arch_results)}


def run_cloud_chip_experiment(experiment, mid_dir, output_dir, num_workers=1, no_cache=False, run_thermal=True):
    records = _cloud_config_records(experiment)
    print("=" * 60)
    print(f"[DSE] cloud {experiment}")
    print("=" * 60)
    tasks = [
        {
            "experiment": experiment,
            "record": record,
            "config_idx": idx,
            "config_total": len(records),
            "mid_dir": mid_dir,
            "output_dir": output_dir,
            "no_cache": no_cache,
            "run_thermal": run_thermal,
        }
        for idx, record in enumerate(records, 1)
    ]
    if num_workers > 1 and len(tasks) > 1:
        print(f"[DSE] launching {len(tasks)} configs with {min(num_workers, len(tasks))} config workers", flush=True)
        with multiprocessing.Pool(min(num_workers, len(tasks)), initializer=set_pdeathsig) as pool:
            results = pool.map(_run_cloud_chip_config, tasks)
    else:
        results = [_run_cloud_chip_config(task) for task in tasks]
    for result in results:
        print(f"[SUMMARY] {result['config']}: {result['num_cases']} cases")
    print(f"[DSE] cloud {experiment} done -> {os.path.join(output_dir, experiment)}")


def _architecture_metric_row(experiment, record, chip_dir):
    chip_path = os.path.join(chip_dir, "chip.yaml")
    if not os.path.exists(chip_path):
        return None
    chip_yaml = load_yaml(chip_path)
    total_area, footprint = validate_chip_area(chip_yaml, core_size)
    logic_power, dram_power = logic_and_dram_power(chip_yaml)
    arch, figure_id = chip_yaml["architecture"], CLOUD_CHIP_EXPERIMENTS[experiment].get("figure_id")
    return {
        "experiment": experiment,
        "config_name": record["name"],
        "config_label": record["label"],
        "order": record["order"],
        "channel_num": record.get("channel_num", ""),
        "sram_mb": record.get("sram_mb", ""),
        "matrix_ratio": record.get("matrix_ratio", ""),
        "noc_flit_B": record.get("noc_flit_B", nested_get(arch, ["noc", "flit_size"], "")),
        "dram_bw_TBps": cloud_dram_bw_tbps(chip_yaml),
        "matrix_compute_TFLOPS": matrix_compute_tflops(chip_yaml, figure_id, record["name"]),
        "vector_compute_TFLOPS": vector_compute_tflops(chip_yaml),
        "sram_area_mm2": to_float(nested_get(arch, ["core", "buffer", "area"])),
        "sram_area_adjusted_mm2": to_float(nested_get(arch, ["core", "buffer", "area"])) - 7.892149,
        "noc_area_mm2": to_float(nested_get(arch, ["noc", "area"])),
        "noc_area_adjusted_mm2": to_float(nested_get(arch, ["noc", "area"])) / 2.0,
        "core_component_area_mm2": total_area,
        "core_footprint_mm2": footprint,
        "logic_power_W": logic_power,
        "dram_power_W": dram_power,
        "max_temperature_c": read_temperature_summary(os.path.join(chip_dir, "thermal")),
        "achieved_compute_ratio_pct": "",
    }


def _case_speedup_rows(experiment, record, chip_dir):
    rows = []
    for case in sort_cloud_cases(read_architecture_summary(os.path.join(chip_dir, "architecture_summary.csv"))):
        baseline = get_h200_baseline(case["model"], case["batch_size"], case["context_length"])
        if not baseline:
            continue
        family = "dense" if case["model"] in CLOUD_DENSE_MODELS else "moe"
        rows.append({
            "experiment": experiment,
            "config_name": record["name"],
            "config_label": record["label"],
            "order": record["order"],
            "model": case["model"],
            "model_label": CLOUD_MODEL_LABELS.get(case["model"], case["model"]),
            "model_family": family,
            "batch_size": case["batch_size"],
            "context_length": case["context_length"],
            "case_label": cloud_case_label(case["model"], case["batch_size"], case["context_length"]),
            "arch_latency_s": case["latency_s"],
            "arch_energy_J": case["energy_J"],
            "h200_latency_s": baseline["latency_s"],
            "h200_energy_J": baseline["energy_J"],
            "speedup": baseline["latency_s"] / case["latency_s"] if case["latency_s"] else 0.0,
            "energy_efficiency": baseline["energy_J"] / case["energy_J"] if case["energy_J"] else 0.0,
            "gemm_latency_s": case.get("gemm_latency_s", 0.0),
            "attention_latency_s": case.get("attention_latency_s", 0.0),
            "communication_latency_s": case.get("intra_comm_latency_s", 0.0) + case.get("inter_comm_latency_s", 0.0),
        })
    return rows


def _average_rows(case_rows):
    rows = []
    config_names = []
    for row in case_rows:
        if row["config_name"] not in config_names:
            config_names.append(row["config_name"])
    for config_name in config_names:
        cfg_rows = [r for r in case_rows if r["config_name"] == config_name]
        lat_weight = sum(to_float(r["arch_latency_s"]) for r in cfg_rows)
        eng_weight = sum(to_float(r["arch_energy_J"]) for r in cfg_rows)
        first = cfg_rows[0]
        rows.append({
            "experiment": first["experiment"],
            "config_name": config_name,
            "config_label": first["config_label"],
            "order": first["order"],
            "avg_speedup": sum(to_float(r["speedup"]) * to_float(r["arch_latency_s"]) for r in cfg_rows) / lat_weight if lat_weight else 0.0,
            "avg_energy_efficiency": sum(to_float(r["energy_efficiency"]) * to_float(r["arch_energy_J"]) for r in cfg_rows) / eng_weight if eng_weight else 0.0,
            "num_cases": len(cfg_rows),
        })
    return sorted(rows, key=lambda r: (to_float(r["order"]), r["config_name"]))


def _operator_rows(case_rows):
    by_case = {}
    for row in case_rows:
        by_case.setdefault((row["model"], row["batch_size"], row["context_length"]), {})[row["config_name"]] = row
    rows = []
    for per_config in by_case.values():
        ordered = sorted(per_config.values(), key=lambda r: (to_float(r["order"]), r["config_name"]))
        if not ordered:
            continue
        gemm_base = ordered[0]
        attn_base = ordered[-1]
        comm_base = next((r for r in ordered if str(r["config_label"]).startswith("32")), ordered[0])
        for row in ordered:
            gemm = to_float(row["gemm_latency_s"])
            attn = to_float(row["attention_latency_s"])
            comm = to_float(row["communication_latency_s"])
            compute = gemm + attn
            total = compute + comm
            rows.append({
                **row,
                "fc_speedup": to_float(gemm_base["gemm_latency_s"]) / gemm if gemm else 0.0,
                "attention_speedup": to_float(attn_base["attention_latency_s"]) / attn if attn else 0.0,
                "communication_speedup": to_float(comm_base["communication_latency_s"]) / comm if comm else 0.0,
                "compute_ratio_pct": compute / total * 100.0 if total else 0.0,
            })
    return rows


def collect_cloud_chip_experiment(experiment, output_dir):
    summary_dir = ensure_dir(os.path.join(output_dir, experiment, "summary"))
    arch_rows = []
    case_rows = []
    for record in _cloud_config_records(experiment):
        chip_dir = _cloud_chip_dir(output_dir, experiment, record["name"])
        arch_row = _architecture_metric_row(experiment, record, chip_dir)
        if arch_row:
            arch_rows.append(arch_row)
        case_rows.extend(_case_speedup_rows(experiment, record, chip_dir))
    avg_rows = _average_rows(case_rows)
    op_rows = _operator_rows(case_rows)

    write_csv_dicts(
        os.path.join(summary_dir, "architecture_metrics.csv"),
        arch_rows,
        [
            "experiment", "config_name", "config_label", "order", "channel_num", "sram_mb",
            "matrix_ratio", "noc_flit_B", "dram_bw_TBps", "matrix_compute_TFLOPS",
            "vector_compute_TFLOPS", "sram_area_mm2", "sram_area_adjusted_mm2",
            "noc_area_mm2", "noc_area_adjusted_mm2", "core_component_area_mm2",
            "core_footprint_mm2", "logic_power_W", "dram_power_W", "max_temperature_c",
            "achieved_compute_ratio_pct",
        ],
    )
    write_csv_dicts(
        os.path.join(summary_dir, "case_speedup.csv"),
        case_rows,
        [
            "experiment", "config_name", "config_label", "order", "model", "model_label",
            "model_family", "batch_size", "context_length", "case_label", "arch_latency_s",
            "arch_energy_J", "h200_latency_s", "h200_energy_J", "speedup",
            "energy_efficiency", "gemm_latency_s", "attention_latency_s", "communication_latency_s",
        ],
    )
    write_csv_dicts(
        os.path.join(summary_dir, "average_speedup.csv"),
        avg_rows,
        ["experiment", "config_name", "config_label", "order", "avg_speedup", "avg_energy_efficiency", "num_cases"],
    )
    write_csv_dicts(
        os.path.join(summary_dir, "operator_analysis.csv"),
        op_rows,
        [
            "experiment", "config_name", "config_label", "order", "model", "model_label",
            "model_family", "batch_size", "context_length", "case_label", "fc_speedup",
            "attention_speedup", "communication_speedup", "compute_ratio_pct",
            "gemm_latency_s", "attention_latency_s", "communication_latency_s",
        ],
    )
    if experiment == "05_baseline_comparison":
        _write_baseline_comparison_csvs(summary_dir, case_rows, avg_rows)
    print(f"[COLLECT] {experiment}: {summary_dir}")


def _write_baseline_comparison_csvs(summary_dir, case_rows, avg_rows):
    rows = []
    for model, bs, ctx in CLOUD_CASE_ORDER:
        rows.append({
            "case_label": cloud_case_label(model, bs, ctx),
            "baseline": "GPU",
            "model": model,
            "model_family": "dense" if model in CLOUD_DENSE_MODELS else "moe",
            "batch_size": bs,
            "context_length": ctx,
            "speedup": 1.0,
            "energy_efficiency": 1.0,
        })
        for row in case_rows:
            if row["model"] == model and int(row["batch_size"]) == bs and int(row["context_length"]) == ctx:
                rows.append({
                    "case_label": row["case_label"],
                    "baseline": "ATLAS" if row["config_name"] == "atlas" else "Stratum",
                    "model": model,
                    "model_family": row["model_family"],
                    "batch_size": bs,
                    "context_length": ctx,
                    "speedup": row["speedup"],
                    "energy_efficiency": row["energy_efficiency"],
                })
    avg = {row["config_name"]: row for row in avg_rows}
    avg_cmp = [
        {"baseline": "GPU", "avg_speedup": 1.0, "avg_energy_efficiency": 1.0},
        {
            "baseline": "Stratum",
            "avg_speedup": to_float(avg.get("stratum", {}).get("avg_speedup")),
            "avg_energy_efficiency": to_float(avg.get("stratum", {}).get("avg_energy_efficiency")),
        },
        {
            "baseline": "ATLAS",
            "avg_speedup": to_float(avg.get("atlas", {}).get("avg_speedup")),
            "avg_energy_efficiency": to_float(avg.get("atlas", {}).get("avg_energy_efficiency")),
        },
    ]
    write_csv_dicts(
        os.path.join(summary_dir, "baseline_case_comparison.csv"),
        rows,
        ["case_label", "baseline", "model", "model_family", "batch_size", "context_length", "speedup", "energy_efficiency"],
    )
    write_csv_dicts(
        os.path.join(summary_dir, "baseline_average_comparison.csv"),
        avg_cmp,
        ["baseline", "avg_speedup", "avg_energy_efficiency"],
    )


def _read_summary_csvs(output_dir, experiment):
    summary_dir = os.path.join(output_dir, experiment, "summary")

    def read(name):
        path = os.path.join(summary_dir, name)
        if not os.path.exists(path):
            return []
        with open(path, "r", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]

    return read("architecture_metrics.csv"), read("average_speedup.csv"), read("case_speedup.csv"), read("operator_analysis.csv")


def _plot_dual_axis(
    ax,
    labels,
    left_values,
    right_values,
    left_label,
    right_label,
    left_ylim,
    left_yticks,
    right_ylim,
    right_yticks,
    left_name,
    right_name,
    left_style,
    right_style,
):
    xs = list(range(len(labels)))
    line_defaults = {
        "markersize": 5.0,
        "markerfacecolor": "white",
        "markeredgewidth": 1.0,
    }
    l1 = ax.plot(xs, left_values, label=left_name, **line_defaults, **left_style)
    ax.set_ylabel(left_label)
    set_paper_y_axis(ax, left_ylim, left_yticks)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax2 = ax.twinx()
    l2 = ax2.plot(xs, right_values, label=right_name, **line_defaults, **right_style)
    ax2.set_ylabel(right_label)
    set_paper_y_axis(ax2, right_ylim, right_yticks)
    lines = l1 + l2
    ax.legend(lines, [line.get_label() for line in lines], loc="best", frameon=False)
    return ax2


def _plot_average_speedup(ax, avg_rows, x_label, ylim, yticks):
    rows = sorted(avg_rows, key=lambda r: to_float(r["order"]))
    ax.bar(
        range(len(rows)),
        [to_float(r["avg_speedup"]) for r in rows],
        color=CHIP_DSE_CONFIG_COLORS[0],
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["config_label"] for r in rows])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Avg. Speedup")
    set_paper_y_axis(ax, ylim, yticks)


def _plot_case_speedup(
    case_rows,
    output_base,
    left_ylim,
    left_yticks,
    right_ylim,
    right_yticks,
    annotate_clipped=False,
):
    import matplotlib.pyplot as plt

    rows = sorted(case_rows, key=lambda r: (to_float(r["order"]), r["config_name"]))
    configs = []
    for row in rows:
        if row["config_label"] not in configs:
            configs.append(row["config_label"])
    cases = list(CLOUD_CASE_ORDER)
    labels = [cloud_case_label(model, bs, ctx) for model, bs, ctx in cases]
    x = list(range(len(cases)))
    width = min(0.75 / max(len(configs), 1), 0.16)
    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    ax2 = ax.twinx()
    ax.set_ylabel("Dense Speedup")
    set_paper_y_axis(ax, left_ylim, left_yticks)
    ax2.set_ylabel("MoE Speedup")
    set_paper_y_axis(ax2, right_ylim, right_yticks)

    def annotate_clipped_values(axis, positions, values, upper_limit):
        inset = (upper_limit - axis.get_ylim()[0]) * 0.025
        for xpos, value in zip(positions, values):
            if value <= upper_limit:
                continue
            axis.text(
                xpos,
                upper_limit - inset,
                rf"${value:.2f}\times$",
                ha="center",
                va="top",
                rotation=90,
                fontsize=6.5,
                clip_on=True,
                zorder=5,
            )

    for idx, cfg in enumerate(configs):
        dense_x, dense_vals, moe_x, moe_vals = [], [], [], []
        for pos, (model, bs, ctx) in enumerate(cases):
            match = next(
                (
                    row for row in rows
                    if row["config_label"] == cfg and row["model"] == model
                    and int(row["batch_size"]) == bs and int(row["context_length"]) == ctx
                ),
                None,
            )
            value = to_float(match["speedup"]) if match else 0.0
            xpos = pos + (idx - (len(configs) - 1) / 2) * width
            if model in CLOUD_DENSE_MODELS:
                dense_x.append(xpos)
                dense_vals.append(value)
            else:
                moe_x.append(xpos)
                moe_vals.append(value)
        color = CHIP_DSE_CONFIG_COLORS[idx % len(CHIP_DSE_CONFIG_COLORS)]
        bar_style = {"color": color, "edgecolor": "black", "linewidth": 0.5}
        ax.bar(dense_x, dense_vals, width=width, label=cfg, **bar_style)
        ax2.bar(moe_x, moe_vals, width=width, **bar_style)
        if annotate_clipped:
            annotate_clipped_values(ax, dense_x, dense_vals, left_ylim[1])
            annotate_clipped_values(ax2, moe_x, moe_vals, right_ylim[1])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(
        ncol=min(5, len(configs)),
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        borderaxespad=0,
    )
    save_basic_figure(fig, output_base)
    plt.close(fig)


def _plot_operator_analysis(
    operator_rows,
    output_base,
    bar_key,
    bar_label,
    bar_ylim,
    bar_yticks,
    line_key,
    line_label,
    line_ylim,
    line_yticks,
):
    import matplotlib.pyplot as plt

    rows = sorted(operator_rows, key=lambda r: (to_float(r["order"]), r["config_name"]))
    configs = []
    for row in rows:
        if row["config_label"] not in configs:
            configs.append(row["config_label"])
    cases = list(CLOUD_CASE_ORDER)
    labels = [cloud_case_label(model, bs, ctx) for model, bs, ctx in cases]
    x = list(range(len(cases)))
    width = min(0.75 / max(len(configs), 1), 0.16)
    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    line_points_by_case = [[] for _ in cases]
    for idx, cfg in enumerate(configs):
        values = []
        offsets = []
        for pos, (model, bs, ctx) in enumerate(cases):
            match = next(
                (
                    row for row in rows
                    if row["config_label"] == cfg and row["model"] == model
                    and int(row["batch_size"]) == bs and int(row["context_length"]) == ctx
                ),
                None,
            )
            values.append(to_float(match[bar_key]) if match else 0.0)
            xpos = pos + (idx - (len(configs) - 1) / 2) * width
            offsets.append(xpos)
            line_points_by_case[pos].append((xpos, to_float(match[line_key]) if match else 0.0))
        ax.bar(
            offsets,
            values,
            width=width,
            label=cfg,
            color=CHIP_DSE_CONFIG_COLORS[idx % len(CHIP_DSE_CONFIG_COLORS)],
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_ylabel(bar_label)
    set_paper_y_axis(ax, bar_ylim, bar_yticks)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax2 = ax.twinx()
    for pos, line_pairs in enumerate(line_points_by_case):
        line_pairs = sorted(line_pairs, key=lambda item: item[0])
        ax2.plot(
            [p[0] for p in line_pairs],
            [p[1] for p in line_pairs],
            marker="o",
            markersize=3.2,
            markerfacecolor="white",
            markeredgewidth=0.8,
            color="black",
            label=line_label if pos == 0 else "_nolegend_",
        )
    ax2.set_ylabel(line_label)
    set_paper_y_axis(ax2, line_ylim, line_yticks)
    handles, legend_labels = ax.get_legend_handles_labels()
    line_handles, line_legend_labels = ax2.get_legend_handles_labels()
    ax.legend(
        handles + line_handles,
        legend_labels + line_legend_labels,
        ncol=min(6, len(configs) + 1),
        frameon=False,
        loc="upper left",
    )
    save_basic_figure(fig, output_base)
    plt.close(fig)


def _remove_legacy_figures(fig_dir, stems):
    for stem in stems:
        for extension in (".pdf", ".png"):
            path = os.path.join(fig_dir, stem + extension)
            if os.path.exists(path):
                os.remove(path)


def plot_cloud_chip_experiment(experiment, output_dir):
    import matplotlib.pyplot as plt

    apply_paper_style()
    arch_rows, avg_rows, case_rows, operator_rows = _read_summary_csvs(output_dir, experiment)
    if not arch_rows and not case_rows:
        raise FileNotFoundError(f"No collected CSVs found for {experiment}; run --action collect first")
    fig_dir = ensure_dir(os.path.join(output_dir, experiment, "figures"))

    if experiment == "01_bandwidth_allocation":
        rows = sorted(arch_rows, key=lambda r: to_float(r["order"]))
        labels = [r["config_label"] for r in rows]
        fig, ax = plt.subplots(figsize=(4.4, 2.7))
        _plot_dual_axis(
            ax, labels,
            [to_float(r["dram_bw_TBps"]) for r in rows],
            [to_float(r["matrix_compute_TFLOPS"]) for r in rows],
            "BW. (TB/s)", "Comp. (TFLOPS)",
            (0, 4.5), (0, 1, 2, 3, 4),
            (0, 18), (0, 4, 8, 12, 16),
            "Bandwidth", "Compute",
            {"color": CHIP_DSE_COLORS["green"], "marker": "^"},
            {"color": CHIP_DSE_COLORS["blue"], "marker": "s"},
        )
        ax.set_xlabel("#Ch.")
        save_basic_figure(fig, os.path.join(fig_dir, "fig13a_bandwidth_compute"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.4, 2.7))
        xs = list(range(len(rows)))
        bars = ax.bar(
            xs,
            [to_float(r["max_temperature_c"]) for r in rows],
            color=CHIP_DSE_COLORS["temperature"],
            edgecolor="black",
            linewidth=0.6,
            label="Temperature",
        )
        threshold = ax.axhline(
            85,
            color=CHIP_DSE_COLORS["threshold_red"],
            linestyle="--",
            linewidth=1.0,
            label="85C Threshold",
        )
        set_paper_y_axis(ax, (0, 126), (0, 30, 60, 90, 120))
        ax.set_ylabel("Temp. (C)")
        ax.set_xlabel("#Ch.")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax2 = ax.twinx()
        compute_line = ax2.plot(
            xs,
            # Due to data confidentiality issue, we need to directly provide results here.
            [100, 100, 100, 70, 5],
            color=CHIP_DSE_COLORS["blue"],
            marker="s",
            markersize=5.0,
            markerfacecolor="white",
            markeredgewidth=1.0,
            label="Achieved Comp.",
        )[0]
        ax2.set_ylabel("Ratio (%)")
        set_paper_y_axis(ax2, (0, 106), (0, 25, 50, 75, 100))
        ax.legend(
            [bars, compute_line, threshold],
            ["Temperature", "Achieved Comp.", "85C Threshold"],
            frameon=False,
            ncol=3,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            borderaxespad=0,
        )
        save_basic_figure(fig, os.path.join(fig_dir, "fig13b_temperature_ratio"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.4, 2.7))
        _plot_average_speedup(ax, avg_rows, "#Ch.", (0, 3), (0, 1, 2, 3))
        save_basic_figure(fig, os.path.join(fig_dir, "fig13c_avg_speedup"))
        plt.close(fig)
        _plot_case_speedup(
            case_rows,
            os.path.join(fig_dir, "fig13d_case_speedup"),
            (0, 4),
            (0, 1, 2, 3, 4),
            (0, 4),
            (0, 1, 2, 3, 4),
            annotate_clipped=True,
        )
        _remove_legacy_figures(
            fig_dir,
            ("fig12a_bandwidth_compute", "fig12b_temperature_ratio", "fig12c_avg_speedup", "fig12d_case_speedup"),
        )

    elif experiment == "02_sram_allocation":
        rows = sorted(arch_rows, key=lambda r: to_float(r["order"]))
        labels = [r["config_label"] for r in rows]
        fig, ax = plt.subplots(figsize=(4.4, 2.7))
        _plot_dual_axis(
            ax, labels,
            [to_float(r["sram_area_adjusted_mm2"]) for r in rows],
            [to_float(r["matrix_compute_TFLOPS"]) for r in rows],
            "Area (mm^2)", "Comp. (TFLOPS)",
            (0, 16), (0, 4, 8, 12, 16),
            (4, 20), (4, 8, 12, 16, 20),
            "SRAM Size", "Compute",
            {"color": CHIP_DSE_COLORS["gray"], "marker": "o"},
            {"color": CHIP_DSE_COLORS["blue"], "marker": "s"},
        )
        ax.set_xlabel("SRAM")
        save_basic_figure(fig, os.path.join(fig_dir, "fig14a_sram_area_compute"))
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(4.4, 2.7))
        _plot_average_speedup(ax, avg_rows, "SRAM", (0, 3), (0, 1, 2, 3))
        save_basic_figure(fig, os.path.join(fig_dir, "fig14b_avg_speedup"))
        plt.close(fig)
        _plot_case_speedup(
            case_rows,
            os.path.join(fig_dir, "fig14c_case_speedup"),
            (0.5, 3.5),
            (0.5, 1.5, 2.5, 3.5),
            (1, 4),
            (1, 2, 3, 4),
        )
        _remove_legacy_figures(fig_dir, ("fig13a_sram_area_compute", "fig13b_avg_speedup", "fig13c_case_speedup"))

    elif experiment == "03_matrix_vector_allocation":
        rows = sorted(arch_rows, key=lambda r: to_float(r["order"]))
        labels = [r["config_label"] for r in rows]
        fig, ax = plt.subplots(figsize=(4.4, 2.7))
        _plot_dual_axis(
            ax, labels,
            [to_float(r["matrix_compute_TFLOPS"]) for r in rows],
            [to_float(r["vector_compute_TFLOPS"]) for r in rows],
            "Matrix (TFLOPS)", "Vector (TFLOPS)",
            (9, 18), (9, 12, 15, 18),
            (0, 3), (0, 1, 2, 3),
            "Matrix Compute", "Vector Compute",
            {"color": CHIP_DSE_COLORS["blue"], "marker": "s"},
            {"color": CHIP_DSE_COLORS["yellow"], "marker": "D"},
        )
        ax.set_xlabel("M. : V.")
        save_basic_figure(fig, os.path.join(fig_dir, "fig15a_matrix_vector_compute"))
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(4.4, 2.7))
        _plot_average_speedup(ax, avg_rows, "M. : V.", (2.0, 2.6), (2.0, 2.2, 2.4, 2.6))
        save_basic_figure(fig, os.path.join(fig_dir, "fig15b_avg_speedup"))
        plt.close(fig)
        _plot_case_speedup(
            case_rows,
            os.path.join(fig_dir, "fig15c_top_case_speedup"),
            (1, 3.5),
            (1, 1.5, 2, 2.5, 3, 3.5),
            (1.5, 4),
            (1.5, 2, 2.5, 3, 3.5, 4),
        )
        _plot_operator_analysis(
            operator_rows, os.path.join(fig_dir, "fig15c_bottom_fc_attention"),
            "fc_speedup", "FC Speedup", (0.9, 1.42), (0.9, 1.0, 1.1, 1.2, 1.3, 1.4),
            "attention_speedup", "Attn. Speedup", (0.9, 1.42), (0.9, 1.0, 1.1, 1.2, 1.3, 1.4),
        )
        _remove_legacy_figures(
            fig_dir,
            ("fig14a_matrix_vector_compute", "fig14b_avg_speedup", "fig14c_top_case_speedup", "fig14c_bottom_fc_attention"),
        )

    elif experiment == "04_noc_allocation":
        rows = sorted(arch_rows, key=lambda r: to_float(r["order"]))
        labels = [r["config_label"] for r in rows]
        fig, ax = plt.subplots(figsize=(4.4, 2.7))
        _plot_dual_axis(
            ax, labels,
            [to_float(r["noc_area_adjusted_mm2"]) for r in rows],
            [to_float(r["matrix_compute_TFLOPS"]) for r in rows],
            "Area (mm^2)", "Comp. (TFLOPS)",
            (0, 16), (0, 4, 8, 12, 16),
            (0, 24), (0, 6, 12, 18, 24),
            "NoC Link Width", "Compute",
            {"color": CHIP_DSE_COLORS["orange"], "marker": "x"},
            {"color": CHIP_DSE_COLORS["blue"], "marker": "s"},
        )
        ax.set_xlabel("Flit")
        save_basic_figure(fig, os.path.join(fig_dir, "fig16a_noc_area_compute"))
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(4.4, 2.7))
        _plot_average_speedup(ax, avg_rows, "Flit", (0, 3), (0, 1, 2, 3))
        save_basic_figure(fig, os.path.join(fig_dir, "fig16b_avg_speedup"))
        plt.close(fig)
        _plot_case_speedup(
            case_rows,
            os.path.join(fig_dir, "fig16c_top_case_speedup"),
            (0, 4),
            (0, 1, 2, 3, 4),
            (0, 4),
            (0, 1, 2, 3, 4),
        )
        _plot_operator_analysis(
            operator_rows, os.path.join(fig_dir, "fig16c_bottom_comm_compute_ratio"),
            "communication_speedup", "Comm. Speedup", (0, 10), (0, 2, 4, 6, 8, 10),
            "compute_ratio_pct", "Comp. Ratio (%)", (0, 100), (0, 20, 40, 60, 80, 100),
        )
        _remove_legacy_figures(
            fig_dir,
            ("fig15a_noc_area_compute", "fig15b_avg_speedup", "fig15c_top_case_speedup", "fig15c_bottom_comm_compute_ratio"),
        )

    elif experiment == "05_baseline_comparison":
        _plot_ppa_figures(output_dir, experiment, arch_rows)
        _plot_fig19_baseline_comparison(output_dir, experiment)
        _remove_legacy_figures(fig_dir, ("table4_ppa_thermal", "fig16_baseline_comparison"))


def _plot_ppa_figures(output_dir, experiment, arch_rows):
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig_dir = ensure_dir(os.path.join(output_dir, experiment, "figures"))
    row_by_name = {r["config_name"]: r for r in arch_rows if r["config_name"] in ("atlas", "stratum")}
    rows = [row_by_name[name] for name in ("atlas", "stratum") if name in row_by_name]
    atlas_total = max([to_float(r["core_component_area_mm2"]) for r in rows if r["config_name"] == "atlas"] or [0.0])
    pie_order = ["Core Ctrl.", "Matrix", "Vector", "SRAM", "NoC", "3D-DRAM/Mem. Ctrl.", "Unused"]
    pie_colors = {
        "Core Ctrl.": "#597CC5",
        "Matrix": "#8AB9D6",
        "Vector": "#A8DBA4",
        "SRAM": "#FDFAB9",
        "NoC": "#AFABAB",
        "3D-DRAM/Mem. Ctrl.": "#F4D8C5",
        "Unused": "#4F5B66",
    }

    def draw_value_legend(ax, labels, values, unit):
        ax.set_axis_off()
        if not labels:
            return
        step = min(0.135, 0.86 / max(len(labels), 1))
        y = 0.94
        for label in labels:
            ax.add_patch(
                Rectangle(
                    (0.02, y - 0.045),
                    0.08,
                    0.045,
                    transform=ax.transAxes,
                    facecolor=pie_colors[label],
                    edgecolor="black",
                    linewidth=0.4,
                )
            )
            ax.text(
                0.14,
                y - 0.023,
                f"{label}\n{values[label]:.2f} {unit}",
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=6.0,
                linespacing=1.05,
            )
            y -= step

    for row in rows:
        fig, axes = plt.subplots(
            1,
            5,
            figsize=(14.0, 3.2),
            gridspec_kw={"width_ratios": [1.0, 0.95, 1.0, 1.05, 1.35]},
        )
        chip_yaml = load_yaml(os.path.join(output_dir, experiment, row["config_name"], "chip.yaml"))
        areas = chip_component_areas(chip_yaml)
        unused = max(0.0, atlas_total - sum(areas.values()))
        if unused:
            areas["Unused"] = unused
        powers = chip_component_powers(chip_yaml)
        for col_idx, (values, title, unit) in enumerate(((areas, "Area", "mm2"), (powers, "Power", "W"))):
            ax = axes[col_idx * 2]
            legend_ax = axes[col_idx * 2 + 1]
            labels = [name for name in pie_order if values.get(name, 0) > 0]
            vals = [values[name] for name in labels]
            colors = [pie_colors[name] for name in labels]
            ax.pie(
                vals,
                colors=colors,
                startangle=90,
                counterclock=False,
                wedgeprops={"edgecolor": "black", "linewidth": 0.4},
            )
            ax.axis("equal")
            ax.set_title(f"{row['config_label']} {title}")
            draw_value_legend(legend_ax, labels, values, unit)
        ax = axes[4]
        thermal_png = os.path.join(output_dir, experiment, row["config_name"], "thermal", "thermal_map.png")
        if os.path.exists(thermal_png):
            ax.imshow(mpimg.imread(thermal_png))
        ax.set_title(f"{row['config_label']} Thermal")
        ax.set_axis_off()
        figure_number = 17 if row["config_name"] == "atlas" else 18
        save_basic_figure(fig, os.path.join(fig_dir, f"fig{figure_number}_{row['config_name']}_ppa_thermal"))
        plt.close(fig)


def _plot_fig19_baseline_comparison(output_dir, experiment):
    import matplotlib.pyplot as plt

    summary_dir = os.path.join(output_dir, experiment, "summary")
    fig_dir = ensure_dir(os.path.join(output_dir, experiment, "figures"))
    with open(os.path.join(summary_dir, "baseline_case_comparison.csv"), "r", newline="") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    with open(os.path.join(summary_dir, "baseline_average_comparison.csv"), "r", newline="") as f:
        avg_rows = [dict(row) for row in csv.DictReader(f)]

    baselines = ["GPU", "Stratum", "ATLAS"]
    cases = list(CLOUD_CASE_ORDER)
    labels = [cloud_case_label(model, bs, ctx) for model, bs, ctx in cases]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [4, 1]})
    axis_rows = (
        (
            "speedup",
            "Dense Speedup", (0, 3.5), (0, 1, 2, 3),
            "MoE Speedup", (0, 7), (0, 2, 4, 6),
            "Avg. Speedup", (0, 3), (0, 1, 2, 3),
        ),
        (
            "energy_efficiency",
            "Dense Efficiency", (0, 12), (0, 3, 6, 9, 12),
            "MoE Efficiency", (0, 12), (0, 3, 6, 9, 12),
            "Avg. Efficiency", (0, 8), (0, 2, 4, 6, 8),
        ),
    )
    for row_idx, axis_row in enumerate(axis_rows):
        (
            metric,
            dense_label, dense_ylim, dense_yticks,
            moe_label, moe_ylim, moe_yticks,
            avg_label, avg_ylim, avg_yticks,
        ) = axis_row
        ax = axes[row_idx][0]
        ax2 = ax.twinx()
        x = list(range(len(cases)))
        width = 0.22
        for idx, baseline in enumerate(baselines):
            dense_x, dense_vals, moe_x, moe_vals = [], [], [], []
            for pos, (model, bs, ctx) in enumerate(cases):
                match = next(
                    (
                        r for r in rows
                        if r["baseline"] == baseline and r["model"] == model
                        and int(float(r["batch_size"])) == bs and int(float(r["context_length"])) == ctx
                    ),
                    None,
                )
                value = to_float(match.get(metric)) if match else 0.0
                xpos = pos + (idx - 1) * width
                if model in CLOUD_DENSE_MODELS:
                    dense_x.append(xpos)
                    dense_vals.append(value)
                else:
                    moe_x.append(xpos)
                    moe_vals.append(value)
            bar_style = {
                "color": CHIP_DSE_BASELINE_COLORS[baseline],
                "edgecolor": "black",
                "linewidth": 0.5,
            }
            ax.bar(dense_x, dense_vals, width=width, label=baseline, **bar_style)
            ax2.bar(moe_x, moe_vals, width=width, **bar_style)
        ax.set_ylabel(dense_label)
        set_paper_y_axis(ax, dense_ylim, dense_yticks)
        ax2.set_ylabel(moe_label)
        set_paper_y_axis(ax2, moe_ylim, moe_yticks)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        if row_idx == 0:
            ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.02), borderaxespad=0)

        ax_avg = axes[row_idx][1]
        avg_values = [to_float(next((r for r in avg_rows if r["baseline"] == b), {}).get(f"avg_{metric}")) for b in baselines]
        ax_avg.bar(
            range(len(baselines)),
            avg_values,
            color=[CHIP_DSE_BASELINE_COLORS[name] for name in baselines],
            edgecolor="black",
            linewidth=0.6,
        )
        ax_avg.set_xticks(range(len(baselines)))
        ax_avg.set_xticklabels(["G.", "S.", "A."])
        ax_avg.set_ylabel(avg_label)
        set_paper_y_axis(ax_avg, avg_ylim, avg_yticks)
    save_basic_figure(fig, os.path.join(fig_dir, "fig19_baseline_comparison"))
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="all", help="experiment ID/alias or all")
    parser.add_argument("--action", default="all", choices=["run", "collect", "plot", "all"])
    parser.add_argument("--mid-dir", default="results/chip_dse_cloud/mid_results")
    parser.add_argument("--output-dir", default="results/chip_dse_cloud")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--skip-thermal", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    for experiment in _cloud_experiment_list(args.experiment):
        if args.action in ("run", "all"):
            run_cloud_chip_experiment(
                experiment,
                args.mid_dir,
                args.output_dir,
                num_workers=args.num_workers,
                no_cache=args.no_cache,
                run_thermal=not args.skip_thermal,
            )
        if args.action in ("collect", "all"):
            collect_cloud_chip_experiment(experiment, args.output_dir)
        if args.action in ("plot", "all"):
            plot_cloud_chip_experiment(experiment, args.output_dir)


if __name__ == "__main__":
    _start_time = time.perf_counter()
    try:
        main()
    finally:
        print(f"[TIMER] chip_dse_cloud.py elapsed: {time.perf_counter() - _start_time:.2f}s")
