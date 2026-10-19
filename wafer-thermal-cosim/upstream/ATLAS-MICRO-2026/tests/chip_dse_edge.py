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

from frontend.hardware_parser import EdgeSystemConfig
from frontend.model_parser import get_model_config_from_hf
from frontend.util import set_pdeathsig
from inference_test_auto import test_edge_inference
from ae_utils import (
    CHIP_DSE_COLORS,
    CHIP_DSE_CONFIG_COLORS,
    EDGE_CASE_ORDER,
    EDGE_MODEL_LABELS,
    apply_paper_style,
    copy_file,
    edge_case_label,
    ensure_dir,
    logic_and_dram_power,
    load_yaml,
    matrix_compute_gflops,
    save_basic_figure,
    set_paper_y_axis,
    sort_edge_cases,
    to_float,
    to_int,
    validate_chip_area,
    vector_compute_gflops,
    write_csv_dicts,
)


core_size = (2.6e-3, 2.6e-3)
core_array_size = (4, 4)
base_frequency = 1000
temperature_threshold = 85
num_dram_layer = 1
cooling_style = "air"
interposer_scaling_factor = 1.05
EDGE_EXPERIMENT = "01_matrix_vector_allocation"

system_dict = {
    "system": {
        "type": "edge",
        "chip": {
            "config_path": None,
            "intra_channel_vector_length": None,
            "channel_num": 8,
            "channel_bandwidth": 25.6,
            "channel_energy": 7,
        },
        "npu": {
            "frequency": 1000,
            "vec_length": 1024,
            "vec_energy": 0.78,
        },
        "kv_cache": {
            "max_context_length": 2097152,
            "block_size": 1024,
        },
    }
}

EDGE_MODEL_CONFIG_LIST = [
    ("opt_6.7b", "configs/models/opt_6.7b.json", 2, [1, 4, 16], [1024, 2048]),
    ("llama3_8b", "configs/models/llama3_8b.json", 2, [1, 4, 16], [1024, 2048]),
    ("palm_8b", "configs/models/palm_8b.json", 2, [1, 4, 16], [1024, 2048]),
]


def classify_edge_operator(op_name):
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


def write_edge_test_case_csv(
    output_dir,
    chip_performance_list,
    accum_latency_dict,
    accum_energy_dict,
    inter_channel_comm_latency_dict,
    inter_channel_comm_energy_dict,
    softmax_latency,
    softmax_energy,
    total_latency,
    total_energy,
    model_config,
):
    ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, "performance.csv")
    category_latency = {"gemm": 0.0, "attention": 0.0, "communication": 0.0}
    category_energy = {"gemm": 0.0, "attention": 0.0, "communication": 0.0}

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "phase", "operator_name", "e2e_latency(s)", "e2e_cycles", "e2e_energy(J)",
            "matrix_cycles", "vector_cycles", "buffer_cycles", "dram_cycles", "noc_cycles",
            "matrix_energy(J)", "vector_energy(J)", "buffer_energy(J)", "dram_energy(J)", "noc_energy(J)",
            "accumulation_latency(s)", "accumulation_energy(J)",
            "inter_channel_comm_latency(s)", "inter_channel_comm_energy(J)",
            "matrix_util", "vector_util", "buffer_util", "dram_util", "noc_util",
        ])
        for phase_name, phase_perf in chip_performance_list:
            frequency_hz = phase_perf.chip_frequency * 1e6
            for op_name, op_stats in phase_perf.operator_stats:
                lat = op_stats.e2e_cycles / frequency_hz
                eng = op_stats.e2e_energy
                cat = classify_edge_operator(op_name)
                writer.writerow(_fmt_row([
                    phase_name, op_name, lat, op_stats.e2e_cycles, eng,
                    op_stats.matrix_cycles, op_stats.vector_cycles, op_stats.buffer_cycles,
                    op_stats.dram_cycles, op_stats.noc_cycles,
                    op_stats.matrix_energy, op_stats.vector_energy, op_stats.buffer_energy,
                    op_stats.dram_energy, op_stats.noc_energy,
                    accum_latency_dict.get(op_name, 0.0), accum_energy_dict.get(op_name, 0.0),
                    inter_channel_comm_latency_dict.get(op_name, 0.0),
                    inter_channel_comm_energy_dict.get(op_name, 0.0),
                    op_stats.matrix_util(), op_stats.vector_util(), op_stats.buffer_util(),
                    op_stats.dram_util(), op_stats.noc_util(),
                ]))
                category_latency[cat] += lat
                category_energy[cat] += eng

        total_accum_lat = sum(accum_latency_dict.values())
        total_accum_eng = sum(accum_energy_dict.values())
        total_inter_ch_lat = sum(inter_channel_comm_latency_dict.values())
        total_inter_ch_eng = sum(inter_channel_comm_energy_dict.values())

        writer.writerow([])
        writer.writerow(["category", "total_latency(s)", "total_energy(J)"])
        writer.writerow(_fmt_row(["gemm", category_latency["gemm"], category_energy["gemm"]]))
        writer.writerow(_fmt_row(["attention", category_latency["attention"], category_energy["attention"]]))
        writer.writerow(_fmt_row(["intra_chip_communication", category_latency["communication"], category_energy["communication"]]))
        writer.writerow(_fmt_row(["accumulation", total_accum_lat, total_accum_eng]))
        writer.writerow(_fmt_row(["softmax", softmax_latency, softmax_energy]))
        writer.writerow(_fmt_row(["inter_channel_communication", total_inter_ch_lat, total_inter_ch_eng]))

        single_layer_lat = total_latency / model_config.num_layers
        single_layer_eng = total_energy / model_config.num_layers
        all_layers_lat = single_layer_lat * model_config.num_layers
        all_layers_eng = single_layer_eng * model_config.num_layers
        writer.writerow([])
        writer.writerow(["", "total_latency(s)", "total_energy(J)"])
        writer.writerow(_fmt_row(["single_layer_total", single_layer_lat, single_layer_eng]))
        writer.writerow(_fmt_row(["all_layers_total", all_layers_lat, all_layers_eng]))

    return {
        "all_layers_latency": all_layers_lat,
        "all_layers_energy": all_layers_eng,
        "single_layer_latency": single_layer_lat,
        "gemm_latency": category_latency["gemm"],
        "attention_latency": category_latency["attention"],
        "intra_comm_latency": category_latency["communication"],
        "accumulation_latency": total_accum_lat,
        "softmax_latency": softmax_latency,
        "inter_channel_comm_latency": total_inter_ch_lat,
        "gemm_energy": category_energy["gemm"],
        "attention_energy": category_energy["attention"],
        "intra_comm_energy": category_energy["communication"],
    }


def write_edge_architecture_summary_csv(output_dir, arch_results):
    ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, "architecture_summary.csv")
    latency_keys = [
        "gemm_latency", "attention_latency", "intra_comm_latency",
        "accumulation_latency", "softmax_latency", "inter_channel_comm_latency",
    ]
    headers = [
        "model", "batch_size", "context_length", "use_online_softmax",
        "all_layers_latency(s)", "all_layers_energy(J)",
        "single_layer_gemm_latency(s)", "single_layer_attention_latency(s)",
        "single_layer_intra_comm_latency(s)",
        "single_layer_accumulation_latency(s)", "single_layer_softmax_latency(s)",
        "single_layer_inter_channel_comm_latency(s)",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for (model_name, bs, ctx, softmax_mode), result in arch_results:
            writer.writerow(_fmt_row(
                [model_name, bs, ctx, softmax_mode, result["all_layers_latency"], result["all_layers_energy"]]
                + [result.get(key, 0.0) for key in latency_keys]
            ))


def _first_number(text, default=0):
    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else default


def _edge_config_records(include_h2llm=False):
    config_dir = "configs/architecture/chip/edge/matrix_vector"
    records = []
    for fname in os.listdir(config_dir):
        if not fname.endswith((".yaml", ".yml")):
            continue
        name = os.path.splitext(fname)[0]
        match = re.search(r"matrix(\d+)_vector(\d+)", name)
        matrix = int(match.group(1)) if match else _first_number(name)
        vector = int(match.group(2)) if match else 1
        records.append({
            "name": name,
            "path": os.path.join(config_dir, fname),
            "order": matrix,
            "label": f"{matrix}:{vector}",
            "is_h2llm": False,
        })
    records = sorted(records, key=lambda r: (r["order"], r["name"]))
    if include_h2llm:
        records.append({
            "name": "h2llm",
            "path": "configs/architecture/chip/edge/h2llm/h2llm.yaml",
            "order": 10**9,
            "label": "H2-LLM",
            "is_h2llm": True,
        })
    return records


def _edge_chip_dir(output_dir, config_name):
    return os.path.join(output_dir, EDGE_EXPERIMENT, config_name)


def _write_edge_system_config(chip_path, chip_output_dir, is_h2llm):
    chip_yaml = load_yaml(chip_path)
    system_config = copy.deepcopy(system_dict)
    system_config["system"]["chip"]["config_path"] = chip_path
    if is_h2llm:
        system_config["system"]["chip"]["intra_channel_vector_length"] = 0
    else:
        vec_num = int(chip_yaml["architecture"]["core"]["vector"]["vec_num"])
        core_num = int(chip_yaml["architecture"]["core_num"])
        system_config["system"]["chip"]["intra_channel_vector_length"] = vec_num * core_num
    system_path = os.path.join(chip_output_dir, "system.yaml")
    with open(system_path, "w") as f:
        yaml.dump(system_config, f, sort_keys=False, default_flow_style=False)
    return system_path


def _run_edge_thermal(config_name, chip_yaml, chip_output_dir):
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


def _run_edge_chip_config(task):
    record = task["record"]
    config_idx = task["config_idx"]
    config_total = task["config_total"]
    chip_output_dir = _edge_chip_dir(task["output_dir"], record["name"])
    chip_intermediate_dir = os.path.join(task["mid_dir"], EDGE_EXPERIMENT, record["name"])
    print(
        f"[{config_idx}/{config_total}] {EDGE_EXPERIMENT}/{record['name']} "
        f"({record['label']})",
        flush=True,
    )
    ensure_dir(chip_output_dir)
    ensure_dir(chip_intermediate_dir)

    chip_path = os.path.join(chip_output_dir, "chip.yaml")
    copy_file(record["path"], chip_path)
    chip_yaml = load_yaml(chip_path)
    validate_chip_area(chip_yaml, core_size)
    system_path = _write_edge_system_config(chip_path, chip_output_dir, record["is_h2llm"])

    thermal_summary = os.path.join(chip_output_dir, "thermal", "thermal_summary.csv")
    if task["run_thermal"] and (task["no_cache"] or not os.path.exists(thermal_summary)):
        _run_edge_thermal(record["name"], chip_yaml, chip_output_dir)

    with open(system_path, "r") as f:
        edge_config = EdgeSystemConfig.from_yaml(yaml.load(f, Loader=yaml.FullLoader))

    arch_results = []
    gemm_cache_dir = ensure_dir(os.path.join(chip_intermediate_dir, "gemm_tiling_cache"))
    total_cases = sum(len(batch_sizes) * len(context_lengths) for _, _, _, batch_sizes, context_lengths in EDGE_MODEL_CONFIG_LIST)
    case_idx = 0
    for model_name, model_config_path, element_size, batch_sizes, context_lengths in EDGE_MODEL_CONFIG_LIST:
        model_config = get_model_config_from_hf(model_name, model_config_path)
        for batch_size in batch_sizes:
            for context_length in context_lengths:
                case_idx += 1
                softmax_label = "on"
                case_name = f"{model_name}_bs{batch_size}_ctx{context_length}_softmax_{softmax_label}"
                case_mid_dir = ensure_dir(os.path.join(chip_intermediate_dir, case_name))
                case_out_dir = ensure_dir(os.path.join(chip_output_dir, case_name))
                pkl_path = os.path.join(case_out_dir, "raw_performance.pkl")
                if not task["no_cache"] and os.path.exists(pkl_path):
                    try:
                        with open(pkl_path, "rb") as f:
                            cached = pickle.load(f)
                        if "case_result" in cached:
                            arch_results.append(((model_name, batch_size, context_length, softmax_label), cached["case_result"]))
                            print(f"  [{case_idx}/{total_cases}] [CACHED] {case_name}", flush=True)
                            continue
                    except Exception as exc:
                        print(f"[WARN] Failed to read cache {pkl_path}: {exc}")

                print(f"  [{case_idx}/{total_cases}] [RUN] {case_name}", flush=True)
                inference_result = test_edge_inference(
                    edge_config=edge_config,
                    model_config=model_config,
                    context_length_list=[context_length] * batch_size,
                    element_size=element_size,
                    intermediate_result_dir=case_mid_dir,
                    gemm_tiling_cache_dir=gemm_cache_dir,
                )
                case_result = write_edge_test_case_csv(
                    case_out_dir,
                    [("intra_chip", inference_result["intra_chip_computation_performance"])],
                    inference_result["intra_channel_accumulation_latency_dict"],
                    inference_result["intra_channel_accumulation_energy_dict"],
                    inference_result["inter_channel_communication_latency_dict"],
                    inference_result["inter_channel_communication_energy_dict"],
                    inference_result["softmax_latency"],
                    inference_result["softmax_energy"],
                    inference_result["latency"],
                    inference_result["energy"],
                    model_config,
                )
                arch_results.append(((model_name, batch_size, context_length, softmax_label), case_result))
                with open(pkl_path, "wb") as f:
                    pickle.dump({"case_result": case_result}, f)
                print(f"  [{case_idx}/{total_cases}] [DONE] {case_name}", flush=True)
    write_edge_architecture_summary_csv(chip_output_dir, arch_results)
    print(f"[DONE] {EDGE_EXPERIMENT}/{record['name']} -> {chip_output_dir}", flush=True)
    return {"config": f"{EDGE_EXPERIMENT}/{record['name']}", "num_cases": len(arch_results)}


def run_edge_chip_experiment(mid_dir, output_dir, num_workers=1, no_cache=False, run_thermal=True):
    records = _edge_config_records(include_h2llm=True)
    print("=" * 60)
    print(f"[DSE] edge {EDGE_EXPERIMENT}")
    print("=" * 60)
    tasks = [
        {
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
            results = pool.map(_run_edge_chip_config, tasks)
    else:
        results = [_run_edge_chip_config(task) for task in tasks]
    for result in results:
        print(f"[SUMMARY] {result['config']}: {result['num_cases']} cases")
    print(f"[DSE] edge {EDGE_EXPERIMENT} done -> {os.path.join(output_dir, EDGE_EXPERIMENT)}")


def _parse_edge_summary(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            model = (row.get("model") or "").strip()
            if not model or model.endswith("_sum") or model == "all_models_sum":
                continue
            rows.append({
                "model": model,
                "batch_size": to_int(row.get("batch_size")),
                "context_length": to_int(row.get("context_length")),
                "latency_s": to_float(row.get("all_layers_latency(s)")),
                "energy_J": to_float(row.get("all_layers_energy(J)")),
            })
    return rows


def collect_edge_chip_experiment(output_dir):
    summary_dir = ensure_dir(os.path.join(output_dir, EDGE_EXPERIMENT, "summary"))
    h2_rows = _parse_edge_summary(os.path.join(_edge_chip_dir(output_dir, "h2llm"), "architecture_summary.csv"))
    h2_baseline = {(r["model"], r["batch_size"], r["context_length"]): r for r in h2_rows}
    arch_rows = []
    case_rows = []

    for record in _edge_config_records(include_h2llm=True):
        chip_dir = _edge_chip_dir(output_dir, record["name"])
        chip_path = os.path.join(chip_dir, "chip.yaml")
        if not os.path.exists(chip_path):
            continue
        chip_yaml = load_yaml(chip_path)
        total_area, footprint = validate_chip_area(chip_yaml, core_size)
        arch_rows.append({
            "config_name": record["name"],
            "config_label": record["label"],
            "order": record["order"],
            "is_h2llm": int(record["is_h2llm"]),
            "matrix_compute_GFLOPS": matrix_compute_gflops(chip_yaml, "fig21a", record["name"]),
            "vector_compute_GFLOPS": vector_compute_gflops(chip_yaml),
            "core_component_area_mm2": total_area,
            "core_footprint_mm2": footprint,
        })
        if record["is_h2llm"]:
            continue
        for case in sort_edge_cases(_parse_edge_summary(os.path.join(chip_dir, "architecture_summary.csv"))):
            baseline = h2_baseline.get((case["model"], case["batch_size"], case["context_length"]))
            if not baseline:
                continue
            case_rows.append({
                "config_name": record["name"],
                "config_label": record["label"],
                "order": record["order"],
                "model": case["model"],
                "model_label": EDGE_MODEL_LABELS.get(case["model"], case["model"]),
                "batch_size": case["batch_size"],
                "context_length": case["context_length"],
                "case_label": edge_case_label(case["model"], case["batch_size"], case["context_length"]),
                "arch_latency_s": case["latency_s"],
                "arch_energy_J": case["energy_J"],
                "h2llm_latency_s": baseline["latency_s"],
                "h2llm_energy_J": baseline["energy_J"],
                "speedup": baseline["latency_s"] / case["latency_s"] if case["latency_s"] else 0.0,
                "energy_efficiency": baseline["energy_J"] / case["energy_J"] if case["energy_J"] else 0.0,
            })

    avg_rows = []
    config_names = []
    for row in case_rows:
        if row["config_name"] not in config_names:
            config_names.append(row["config_name"])
    for config_name in config_names:
        rows = [r for r in case_rows if r["config_name"] == config_name]
        lat_weight = sum(to_float(r["h2llm_latency_s"]) for r in rows)
        eng_weight = sum(to_float(r["h2llm_energy_J"]) for r in rows)
        first = rows[0]
        avg_rows.append({
            "config_name": config_name,
            "config_label": first["config_label"],
            "order": first["order"],
            "avg_speedup": sum(to_float(r["speedup"]) * to_float(r["h2llm_latency_s"]) for r in rows) / lat_weight if lat_weight else 0.0,
            "avg_energy_efficiency": sum(to_float(r["energy_efficiency"]) * to_float(r["h2llm_energy_J"]) for r in rows) / eng_weight if eng_weight else 0.0,
            "num_cases": len(rows),
        })

    write_csv_dicts(
        os.path.join(summary_dir, "architecture_metrics.csv"),
        arch_rows,
        ["config_name", "config_label", "order", "is_h2llm", "matrix_compute_GFLOPS", "vector_compute_GFLOPS", "core_component_area_mm2", "core_footprint_mm2"],
    )
    write_csv_dicts(
        os.path.join(summary_dir, "case_speedup.csv"),
        case_rows,
        ["config_name", "config_label", "order", "model", "model_label", "batch_size", "context_length", "case_label", "arch_latency_s", "arch_energy_J", "h2llm_latency_s", "h2llm_energy_J", "speedup", "energy_efficiency"],
    )
    write_csv_dicts(
        os.path.join(summary_dir, "average_speedup.csv"),
        avg_rows,
        ["config_name", "config_label", "order", "avg_speedup", "avg_energy_efficiency", "num_cases"],
    )
    print(f"[COLLECT] {EDGE_EXPERIMENT}: {summary_dir}")


def _read_edge_summary_csvs(output_dir):
    summary_dir = os.path.join(output_dir, EDGE_EXPERIMENT, "summary")

    def read(name):
        path = os.path.join(summary_dir, name)
        if not os.path.exists(path):
            return []
        with open(path, "r", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]

    return read("architecture_metrics.csv"), read("average_speedup.csv"), read("case_speedup.csv")


def plot_edge_chip_experiment(output_dir):
    import matplotlib.pyplot as plt

    apply_paper_style()
    arch_rows, avg_rows, case_rows = _read_edge_summary_csvs(output_dir)
    if not arch_rows and not case_rows:
        raise FileNotFoundError(f"No collected CSVs found for {EDGE_EXPERIMENT}; run --action collect first")
    fig_dir = ensure_dir(os.path.join(output_dir, EDGE_EXPERIMENT, "figures"))

    rows = sorted(arch_rows, key=lambda r: to_float(r["order"]))
    labels = [r["config_label"] for r in rows]
    xs = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    line_style = {
        "markersize": 5.0,
        "markerfacecolor": "white",
        "markeredgewidth": 1.0,
    }
    l1 = ax.plot(
        xs,
        [to_float(r["matrix_compute_GFLOPS"]) for r in rows],
        color=CHIP_DSE_COLORS["blue"],
        marker="s",
        label="Matrix Compute",
        **line_style,
    )
    ax.set_ylabel("Matrix (GFLOPS)")
    set_paper_y_axis(ax, (80, 320), (80, 160, 240, 320))
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("M. : V.")
    ax2 = ax.twinx()
    l2 = ax2.plot(
        xs,
        [to_float(r["vector_compute_GFLOPS"]) for r in rows],
        color=CHIP_DSE_COLORS["yellow"],
        marker="D",
        label="Vector Compute",
        **line_style,
    )
    ax2.set_ylabel("Vector (GFLOPS)")
    set_paper_y_axis(ax2, (0, 120), (0, 40, 80, 120))
    lines = l1 + l2
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, loc="best")
    save_basic_figure(fig, os.path.join(fig_dir, "fig21a_matrix_vector_compute"))
    plt.close(fig)

    avg_rows = sorted(avg_rows, key=lambda r: to_float(r["order"]))
    labels = [r["config_label"] for r in avg_rows]
    xs = list(range(len(avg_rows)))
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    bars = ax.bar(
        xs,
        [to_float(r["avg_speedup"]) for r in avg_rows],
        color=CHIP_DSE_CONFIG_COLORS[0],
        edgecolor="black",
        linewidth=0.6,
        label="Speedup",
    )
    ax.set_ylabel("Avg. Speedup")
    set_paper_y_axis(ax, (0.6, 1.2), (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2))
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax2 = ax.twinx()
    efficiency_line = ax2.plot(
        xs,
        [to_float(r["avg_energy_efficiency"]) for r in avg_rows],
        color="#E28E8E",
        marker="o",
        markersize=5.0,
        markerfacecolor="white",
        markeredgewidth=1.0,
        label="Energy Efficiency",
    )[0]
    ax2.set_ylabel("Avg. Efficiency")
    set_paper_y_axis(ax2, (0.6, 1.2), (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2))
    ax.legend(
        [bars, efficiency_line],
        ["Speedup", "Energy Efficiency"],
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        borderaxespad=0,
    )
    save_basic_figure(fig, os.path.join(fig_dir, "fig21b_avg_speedup_efficiency"))
    plt.close(fig)

    configs = [r["config_label"] for r in avg_rows]
    cases = list(EDGE_CASE_ORDER)
    case_labels = [edge_case_label(model, bs, ctx) for model, bs, ctx in cases]
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    width = min(0.75 / max(len(configs), 1), 0.15)
    x = list(range(len(cases)))
    for ax, metric, ylabel, ylim, yticks in (
        (axes[0], "speedup", "Speedup", (0.4, 1.2), (0.4, 0.6, 0.8, 1.0, 1.2)),
        (axes[1], "energy_efficiency", "Efficiency", (0.4, 1.3), (0.4, 0.7, 1.0, 1.3)),
    ):
        for idx, cfg in enumerate(configs):
            values = []
            for model, bs, ctx in cases:
                match = next(
                    (
                        row for row in case_rows
                        if row["config_label"] == cfg and row["model"] == model
                        and int(row["batch_size"]) == bs and int(row["context_length"]) == ctx
                    ),
                    None,
                )
                values.append(to_float(match.get(metric)) if match else 0.0)
            offsets = [pos + (idx - (len(configs) - 1) / 2) * width for pos in x]
            ax.bar(
                offsets,
                values,
                width=width,
                label=f"Mat.:Vec.={cfg}",
                color=CHIP_DSE_CONFIG_COLORS[idx % len(CHIP_DSE_CONFIG_COLORS)],
                edgecolor="black",
                linewidth=0.5,
            )
        ax.set_ylabel(ylabel)
        set_paper_y_axis(ax, ylim, yticks)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(case_labels, rotation=45, ha="right")
    axes[0].legend(frameon=False, ncol=min(5, len(configs)), loc="upper left")
    save_basic_figure(fig, os.path.join(fig_dir, "fig21c_case_speedup_efficiency"))
    plt.close(fig)

    for stem in ("fig18a_matrix_vector_compute", "fig18b_avg_speedup_efficiency", "fig18c_case_speedup_efficiency"):
        for extension in (".pdf", ".png"):
            legacy_path = os.path.join(fig_dir, stem + extension)
            if os.path.exists(legacy_path):
                os.remove(legacy_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 4 AE edge chip DSE")
    parser.add_argument("--experiment", default=EDGE_EXPERIMENT, choices=[EDGE_EXPERIMENT, "matrix_vector", "fig21", "21", "all"])
    parser.add_argument("--action", default="all", choices=["run", "collect", "plot", "all"])
    parser.add_argument("--mid-dir", default="results/chip_dse_edge/mid_results")
    parser.add_argument("--output-dir", default="results/chip_dse_edge")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--skip-thermal", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.action in ("run", "all"):
        run_edge_chip_experiment(
            args.mid_dir,
            args.output_dir,
            num_workers=args.num_workers,
            no_cache=args.no_cache,
            run_thermal=not args.skip_thermal,
        )
    if args.action in ("collect", "all"):
        collect_edge_chip_experiment(args.output_dir)
    if args.action in ("plot", "all"):
        plot_edge_chip_experiment(args.output_dir)


if __name__ == "__main__":
    _start_time = time.perf_counter()
    try:
        main()
    finally:
        print(f"[TIMER] chip_dse_edge.py elapsed: {time.perf_counter() - _start_time:.2f}s")
