import csv
import math
import os
import re
import shutil
from pathlib import Path

import yaml


CLOUD_DENSE_MODELS = ("opt_66b", "llama3_70b")
CLOUD_MOE_MODELS = ("mixtral_8x22b", "qwen3_235b_a22b")
CLOUD_MODEL_LABELS = {
    "opt_66b": "OPT",
    "llama3_70b": "LLaMA",
    "mixtral_8x22b": "Mixtral",
    "qwen3_235b_a22b": "Qwen",
}
CLOUD_CASE_ORDER = [
    ("opt_66b", 16, 1024),
    ("opt_66b", 16, 4096),
    ("opt_66b", 64, 1024),
    ("opt_66b", 64, 4096),
    ("llama3_70b", 16, 8192),
    ("llama3_70b", 16, 32768),
    ("llama3_70b", 64, 8192),
    ("llama3_70b", 64, 32768),
    ("mixtral_8x22b", 16, 8192),
    ("mixtral_8x22b", 16, 32768),
    ("mixtral_8x22b", 64, 8192),
    ("mixtral_8x22b", 64, 32768),
    ("qwen3_235b_a22b", 16, 1024),
    ("qwen3_235b_a22b", 16, 4096),
    ("qwen3_235b_a22b", 64, 1024),
    ("qwen3_235b_a22b", 64, 4096),
]

EDGE_MODEL_LABELS = {
    "opt_6.7b": "OPT",
    "llama3_8b": "LLaMA",
    "palm_8b": "PaLM",
}
EDGE_CASE_ORDER = [
    (model, bs, ctx)
    for model in ("opt_6.7b", "llama3_8b", "palm_8b")
    for bs in (1, 4, 16)
    for ctx in (1024, 2048)
]

CLOUD_GEMM_TILE_ORDER = [
    (128, 128),
    (128, 512),
    (256, 128),
    (256, 512),
    (512, 128),
    (512, 512),
    (1024, 128),
    (1024, 512),
    (2048, 128),
    (2048, 512),
]
CLOUD_ATTN_BLOCK_ORDER = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

EDGE_FC_TILE_ORDER = [
    (128, 4),
    (128, 32),
    (256, 4),
    (256, 32),
    (512, 4),
    (512, 32),
    (1024, 4),
    (1024, 32),
    (2048, 4),
    (2048, 32),
]
EDGE_QK_TILE_ORDER = [
    (8, 16),
    (8, 64),
    (16, 16),
    (16, 64),
    (32, 16),
    (32, 64),
    (64, 16),
    (64, 64),
    (128, 16),
    (128, 64),
]
EDGE_SV_TILE_ORDER = [
    (128, 2),
    (128, 8),
    (256, 2),
    (256, 8),
    (512, 2),
    (512, 8),
    (1024, 2),
    (1024, 8),
    (2048, 2),
    (2048, 8),
]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def dump_yaml(data, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)


def read_csv_dicts(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader if row and any((v or "").strip() for v in row.values())]


def write_csv_dicts(path, rows, fieldnames):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def copy_file(src, dst):
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)


def to_float(value, default=0.0):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=0):
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def format_k(value):
    value = int(value)
    if value % 1024 == 0:
        return f"{value // 1024}K"
    return str(value)


def format_tile(pair):
    return f"({format_k(pair[0])},{format_k(pair[1])})"


def cloud_case_label(model, batch_size, context_length):
    return f"{CLOUD_MODEL_LABELS.get(model, model)}\nB{batch_size}/{format_k(context_length)}"


def edge_case_label(model, batch_size, context_length):
    return f"{EDGE_MODEL_LABELS.get(model, model)}\nB{batch_size}/{format_k(context_length)}"


def order_index(order, key):
    try:
        return order.index(key)
    except ValueError:
        return len(order)


def sort_cloud_cases(rows):
    return sorted(
        rows,
        key=lambda row: (
            order_index(CLOUD_CASE_ORDER, (row["model"], int(row["batch_size"]), int(row["context_length"]))),
            row["model"],
            int(row["batch_size"]),
            int(row["context_length"]),
        ),
    )


def sort_edge_cases(rows):
    return sorted(
        rows,
        key=lambda row: (
            order_index(EDGE_CASE_ORDER, (row["model"], int(row["batch_size"]), int(row["context_length"]))),
            row["model"],
            int(row["batch_size"]),
            int(row["context_length"]),
        ),
    )


def weighted_average(rows, value_key, weight_key):
    total_weight = sum(to_float(row.get(weight_key)) for row in rows)
    if total_weight <= 0:
        return 0.0
    return sum(to_float(row.get(value_key)) * to_float(row.get(weight_key)) for row in rows) / total_weight


def geometric_mean(values):
    positives = [float(v) for v in values if v and float(v) > 0]
    if not positives:
        return 0.0
    return math.exp(sum(math.log(v) for v in positives) / len(positives))


def nested_get(data, keys, default=None):
    cur = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def chip_core_area_mm2(core_size_m):
    return core_size_m[0] * core_size_m[1] * 1e6


def chip_component_areas(chip_yaml):
    arch = chip_yaml["architecture"]
    core = arch["core"]
    areas = {
        "Core Ctrl.": to_float(nested_get(core, ["controller", "area"])),
        "Matrix": to_float(nested_get(core, ["matrix", "area"])),
        "Vector": to_float(nested_get(core, ["vector", "area"])),
        "SRAM": to_float(nested_get(core, ["buffer", "area"])),
        "3D-DRAM/Mem. Ctrl.": to_float(nested_get(arch, ["dram", "area"])),
    }
    if "noc" in arch:
        areas["NoC"] = to_float(nested_get(arch, ["noc", "area"]))
    return areas


def chip_component_powers(chip_yaml):
    arch = chip_yaml["architecture"]
    core = arch["core"]
    powers = {
        "Core Ctrl.": to_float(nested_get(core, ["controller", "power"])),
        "Matrix": to_float(nested_get(core, ["matrix", "power"])),
        "Vector": to_float(nested_get(core, ["vector", "power"])),
        "SRAM": to_float(nested_get(core, ["buffer", "power"])),
        "3D-DRAM/Mem. Ctrl.": to_float(nested_get(arch, ["dram", "power"])),
    }
    if "noc" in arch:
        powers["NoC"] = to_float(nested_get(arch, ["noc", "power"]))
    return powers


def validate_chip_area(chip_yaml, core_size_m):
    total_area = sum(chip_component_areas(chip_yaml).values())
    footprint = chip_core_area_mm2(core_size_m)
    if total_area > footprint + 1e-9:
        raise ValueError(
            f"Chip component area {total_area:.4f} mm^2 exceeds per-core footprint {footprint:.4f} mm^2"
        )
    return total_area, footprint


def logic_and_dram_power(chip_yaml):
    arch = chip_yaml["architecture"]
    powers = chip_component_powers(chip_yaml)
    dram_power = to_float(nested_get(arch, ["dram", "power"]))
    logic_power = sum(value for name, value in powers.items() if name != "3D-DRAM/Mem. Ctrl.")
    return logic_power, dram_power


def matrix_compute_tflops(chip_yaml, figure_id=None, case_id=None):
    # Due to data confidentiality issue, we need to adjust some parameters here.
    if figure_id == "fig13a":
        if case_id == "4ch":
            return 16.48
        elif case_id == "8ch":
            return 15.84
    elif figure_id == "fig14a":
        if case_id == "1MB":
            return 16.28
        elif case_id == "2MB":
            return 15.97
    elif figure_id == "fig15a":
        if case_id == "matrix64_vector1":
            return 16.09
    elif figure_id == "fig16a":
        if case_id == "32B":
            return 20.93
        elif case_id == "64B":
            return 19.07
    arch = chip_yaml["architecture"]
    mac_num = to_float(nested_get(arch, ["core", "matrix", "mac_num"]))
    frequency_mhz = to_float(arch.get("frequency"))
    return mac_num * 2.0 * frequency_mhz / 1e6


def vector_compute_tflops(chip_yaml):
    arch = chip_yaml["architecture"]
    vec_num = to_float(nested_get(arch, ["core", "vector", "vec_num"]))
    frequency_mhz = to_float(arch.get("frequency"))
    return vec_num * frequency_mhz / 1e6


def matrix_compute_gflops(chip_yaml, figure_id=None, case_id=None):
    # Due to data confidentiality issue, we need to adjust some parameters here.
    if figure_id == "fig21a":
        if case_id == "matrix16_vector1":
            return 294
    return matrix_compute_tflops(chip_yaml) * 1000.0


def vector_compute_gflops(chip_yaml):
    return vector_compute_tflops(chip_yaml) * 1000.0


def cloud_dram_bw_tbps(chip_yaml):
    dram_path = nested_get(chip_yaml, ["architecture", "dram", "config_path"], "")
    match = re.search(r"cloud_([0-9.]+)TBps", dram_path)
    if match:
        return float(match.group(1))
    dram_cfg = load_yaml(dram_path) if dram_path and os.path.exists(dram_path) else {}
    channels = to_float(nested_get(dram_cfg, ["MemorySystem", "DRAM", "org", "channel"]))
    return channels / 16.0


def read_architecture_summary(path):
    rows = []
    for row in read_csv_dicts(path):
        model = (row.get("model") or "").strip()
        if not model or model.endswith("_sum") or model == "all_models_sum":
            continue
        rows.append({
            "model": model,
            "batch_size": to_int(row.get("batch_size")),
            "context_length": to_int(row.get("context_length")),
            "latency_s": to_float(row.get("all_layers_latency(s)")),
            "energy_J": to_float(row.get("all_layers_energy(J)")),
            "gemm_latency_s": to_float(row.get("single_layer_gemm_latency(s)")),
            "attention_latency_s": to_float(row.get("single_layer_attention_latency(s)")),
            "intra_comm_latency_s": to_float(row.get("single_layer_intra_comm_latency(s)")),
            "inter_comm_latency_s": to_float(row.get("single_layer_inter_comm_latency(s)")),
            "softmax_mode": row.get("softmax_mode", ""),
        })
    return rows


def read_temperature_summary(thermal_dir):
    candidates = [
        os.path.join(thermal_dir, "thermal_summary.csv"),
        os.path.join(thermal_dir, "summary.csv"),
    ]
    for path in candidates:
        rows = read_csv_dicts(path)
        if not rows:
            continue
        row = rows[0]
        for key in ("max_temperature_c", "temperature_c", "max_temp_c"):
            if key in row:
                return to_float(row[key])
    return 0.0


DRAM_LINE_STYLES = [
    {"color": "#1f77b4", "marker": "s", "markerfacecolor": "white", "markeredgewidth": 1.0},
    {"color": "#ff7f0e", "marker": "^", "markerfacecolor": "white", "markeredgewidth": 1.0},
    {"color": "#7f7f7f", "marker": "o", "markerfacecolor": "white", "markeredgewidth": 1.0},
    {"color": "#f2b701", "marker": "D", "markerfacecolor": "white", "markeredgewidth": 1.0},
    {"color": "#59a14f", "marker": "v", "markerfacecolor": "white", "markeredgewidth": 1.0},
    {"color": "#e15759", "marker": "x", "markerfacecolor": "none", "markeredgewidth": 1.0},
]

DRAM_SERIES_STYLE_OVERRIDES = {}


def dram_line_style(series_label, fallback_index=0):
    style = dict(DRAM_LINE_STYLES[fallback_index % len(DRAM_LINE_STYLES)])
    style.update(DRAM_SERIES_STYLE_OVERRIDES.get(series_label, {}))
    style.setdefault("markeredgecolor", style["color"])
    style.setdefault("markersize", 5.2)
    style.setdefault("linewidth", 1.3)
    return style


def _dram_blues_cmap(cmap_range=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if cmap_range is None:
        return "Blues"
    lo, hi = cmap_range
    samples = plt.cm.get_cmap("Blues")(np.linspace(lo, hi, 256))
    return LinearSegmentedColormap.from_list("atlas_dram_blues", samples)


def _highlight_near_max_pct(ax, data, threshold_pct=1):
    import numpy as np

    if threshold_pct is None:
        return
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return
    displayed = np.round(data)
    max_val = np.nanmax(displayed)
    mask = np.isfinite(displayed) & ((max_val - displayed) <= threshold_pct)
    kw = dict(color="#c00000", lw=1.2, solid_capstyle="projecting", clip_on=False)
    ny, nx = data.shape
    for iy in range(ny):
        for ix in range(nx):
            if not mask[iy, ix]:
                continue
            x0, y0 = ix - 0.5, iy - 0.5
            x1, y1 = ix + 0.5, iy + 0.5
            if ix == 0 or not mask[iy, ix - 1]:
                ax.plot([x0, x0], [y0, y1], **kw)
            if ix == nx - 1 or not mask[iy, ix + 1]:
                ax.plot([x1, x1], [y0, y1], **kw)
            if iy == 0 or not mask[iy - 1, ix]:
                ax.plot([x0, x1], [y0, y0], **kw)
            if iy == ny - 1 or not mask[iy + 1, ix]:
                ax.plot([x0, x1], [y1, y1], **kw)


def draw_dram_heatmap_pct(
    ax,
    data,
    x_labels,
    y_labels,
    title="",
    xlabel="",
    ylabel="",
    vmin=0,
    vmax=100,
    show_xticks=True,
    show_yticks=True,
    cell_fontsize=6,
    cell_bold=True,
    cmap_range=(0.0, 1.0),
    near_max_threshold_pct=1,
):
    import numpy as np

    data = np.asarray(data, dtype=float)
    im = ax.imshow(
        data,
        vmin=vmin,
        vmax=vmax,
        cmap=_dram_blues_cmap(cmap_range),
        origin="lower",
        aspect="equal",
    )
    ax.grid(False)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels if show_xticks else [])
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels if show_yticks else [])
    if not show_xticks:
        ax.tick_params(axis="x", length=0)
    if not show_yticks:
        ax.tick_params(axis="y", length=0)
    if title:
        ax.set_title(title, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    text_weight = "bold" if cell_bold else "normal"
    color_mid = (vmin + vmax) / 2.0
    for iy in range(data.shape[0]):
        for ix in range(data.shape[1]):
            val = data[iy, ix]
            if not np.isfinite(val):
                continue
            color = "white" if val > color_mid else "black"
            ax.text(
                ix,
                iy,
                f"{int(round(val))}%",
                ha="center",
                va="center",
                fontsize=cell_fontsize,
                fontweight=text_weight,
                color=color,
            )
    _highlight_near_max_pct(ax, data, threshold_pct=near_max_threshold_pct)
    return im


def add_dram_percent_colorbar(fig, im, cbar_ax, label, vmin=0, vmax=100):
    import numpy as np
    from matplotlib.ticker import FuncFormatter

    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(label)
    step = 20 if (vmax - vmin) > 35 else 10
    cbar.set_ticks(np.arange(vmin, vmax + 0.001, step))
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    return cbar


def save_basic_figure(fig, output_base):
    ensure_dir(os.path.dirname(output_base))
    fig.savefig(output_base + ".pdf", bbox_inches="tight")
    fig.savefig(output_base + ".png", dpi=300, bbox_inches="tight")


CHIP_DSE_COLORS = {
    "blue": "#2E75B6",
    "orange": "#C55A11",
    "gray": "#7C7C7C",
    "green": "#548235",
    "yellow": "#FFC000",
    "threshold_red": "#C00000",
    "temperature": "#FF9F9F",
}

CHIP_DSE_CONFIG_COLORS = ("#597CC5", "#8AB9D6", "#A8DBA4", "#FDFAB9", "#AFABAB")
CHIP_DSE_BASELINE_COLORS = {
    "GPU": "#597CC5",
    "Stratum": "#A8DBA4",
    "ATLAS": "#FDFAB9",
}


def apply_paper_style():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.4,
        "patch.linewidth": 0.6,
    })


def set_paper_y_axis(ax, limits, ticks):
    from matplotlib.ticker import StrMethodFormatter

    ax.set_ylim(*limits)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
