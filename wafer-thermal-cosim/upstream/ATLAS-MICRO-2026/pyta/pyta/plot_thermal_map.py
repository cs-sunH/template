import os
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def read_heatmap(temperature_file, n_row, n_col, layer_idx=None, t_delta=273.15):
    """
        Read the temperature grid file from hotspot
    """
    with open(temperature_file, 'r') as f:
        raw_lines = f.readlines()

    # Select lines corresponding to the target layer (Hotspot writes "Layer k:" headers)
    if layer_idx is not None:
        header = f"Layer {layer_idx}:"
        target_line_idx = None
        for idx, line in enumerate(raw_lines):
            if line.strip() == header:
                target_line_idx = idx
                break
        if target_line_idx is None:
            raise ValueError(f"Layer {layer_idx} not found in {temperature_file}")
        # Collect lines until the next "Layer" header or EOF
        lines = []
        for line in raw_lines[target_line_idx + 1 :]:
            if line.strip().startswith("Layer "):
                break
            if line.strip():
                lines.append(line)
    else:
        # Use all non-empty, non-header lines
        lines = [line for line in raw_lines if line.strip() and not line.strip().startswith("Layer ")]

    # Each data line is "<index> <temp>"
    values = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        values.append(float(parts[1]))

    if len(values) < n_row * n_col:
        raise ValueError(
            f"Not enough temperature samples for a {n_row}x{n_col} grid: "
            f"got {len(values)} values."
        )

    temperature = np.zeros((n_row, n_col))

    for i in range(n_row):
        for j in range(n_col):
            temperature[i][j] = values[i * n_col + j] - t_delta

    # debug: print max and min temperature
    print(f"Max temperature: {np.max(temperature)}")
    print(f"Min temperature: {np.min(temperature)}")

    return temperature


def read_floorplan(floorplan_file, n_row, n_col):
    """
        Read the floorplan file
    """
    units = []

    max_height = 0
    max_width = 0

    with open(floorplan_file, 'r') as file:
        for line in file:
            if line.strip() == '' or line.strip().startswith('#'):
                continue

            # the size should be normalized according to the grid size
            parts = line.strip().split()
            unit = {
                'unit_name': parts[0],
                'width': float(parts[1]),
                'height': float(parts[2]),
                'left_x': float(parts[3]),
                'bottom_y': float(parts[4]),
                'specific_heat': None,
                'resistivity': None
            }

            # update the max height and width
            max_height = max(max_height, unit['height'] + unit['bottom_y'])
            max_width = max(max_width, unit['width'] + unit['left_x'])

            # only consider chiplets
            if not unit['unit_name'].startswith('Chiplet'):
                continue
            units.append(unit)

    # normalize the coordinates
    for unit in units:
        unit['left_x'] = unit['left_x'] / max_width * n_col
        unit['bottom_y'] = unit['bottom_y'] / max_height * n_row
        unit['width'] = unit['width'] / max_width * n_col
        unit['height'] = unit['height'] / max_height * n_row

    return units


def plot_multi_die_heatmap(
    heatmap,
    floorplan,
    tmin=None,
    tmax=None,
    # Matplotlib plotting parameters.
    figure_size=(5, 4),
    heatmap_axes_rect=(0.01, 0.01, 0.78, 0.98),
    colorbar_x=0.82,
    colorbar_width=0.04,
    colorbar_bottom_margin=0.02,
    colorbar_top_margin=0.02,
    colorbar_ticks=(45, 55, 65, 75, 85),
    colorbar_tick_pad=2,
    colorbar_tick_font_size=18,
    colorbar_label_font_size=18,
    max_temp_annotation_font_size=18,
    annotation_axes_margin=0.05,
):
    """
    Plot the heatmap and overlay the chiplets' outlines
    """
    n_row, n_col = heatmap.shape
    chiplets = floorplan
    max_row, max_col = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    max_temp = heatmap[max_row, max_col]
    max_x = max_col + 0.5
    max_y = max_row + 0.5
    text_x = annotation_axes_margin if max_x > n_col * 0.5 else 1.0 - annotation_axes_margin
    text_y = annotation_axes_margin if max_y > n_row * 0.5 else 1.0 - annotation_axes_margin
    horizontal_alignment = 'left' if text_x < 0.5 else 'right'
    vertical_alignment = 'bottom' if text_y < 0.5 else 'top'

    fig = plt.figure(figsize=figure_size)
    ax = fig.add_axes(heatmap_axes_rect)

    # Plot the heatmap
    im = ax.imshow(heatmap, cmap='plasma', origin='lower', extent=[0, n_col, 0, n_row], vmin=tmin, vmax=tmax)
    heatmap_pos = ax.get_position()
    colorbar_bottom = heatmap_pos.y0 + heatmap_pos.height * colorbar_bottom_margin
    colorbar_height = heatmap_pos.height * (1.0 - colorbar_bottom_margin - colorbar_top_margin)
    cax = fig.add_axes([colorbar_x, colorbar_bottom, colorbar_width, colorbar_height])
    cbar = fig.colorbar(im, cax=cax)
    if colorbar_ticks is not None:
        cbar.set_ticks(colorbar_ticks)
    cbar.ax.tick_params(labelsize=colorbar_tick_font_size, pad=colorbar_tick_pad)
    cbar.set_label('Temperature (°C)', size=colorbar_label_font_size)
    tick_labels = cbar.ax.get_yticklabels()
    if tick_labels:
        tick_labels[0].set_verticalalignment('bottom')
        tick_labels[-1].set_verticalalignment('top')

    # Overlay the chiplets' outlines
    for chiplet in chiplets:
        left_x = chiplet['left_x']
        bottom_y = chiplet['bottom_y']
        width = chiplet['width']
        height = chiplet['height']

        rect = patches.Rectangle(
            (left_x, bottom_y), width, height,
            linewidth=1, edgecolor='white', facecolor='none'
        )
        ax.add_patch(rect)

    ax.scatter(max_x, max_y, s=70, c='darkred', edgecolors='white', linewidths=1.0, zorder=5)
    ax.annotate(
        f"Max: {max_temp:.1f}℃",
        xy=(max_x, max_y),
        xytext=(text_x, text_y),
        textcoords='axes fraction',
        color='white',
        fontsize=max_temp_annotation_font_size,
        ha=horizontal_alignment,
        va=vertical_alignment,
        multialignment='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='darkred', edgecolor='white', alpha=0.85),
        arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
        zorder=6,
        annotation_clip=False,
    )

    # Set labels and title
    # ax.set_xlabel('X Position')
    # ax.set_ylabel('Y Position')
    # ax.set_title('Heatmap with Chiplet Outlines (T_max = %.2f°C)' % np.max(heatmap))

    # remove x/y ticks
    ax.set_xticks([])
    ax.set_yticks([])


def plot_hotspot_thermal_map(
    output_dir,
    n_row=128,
    n_col=128,
    t_delta=273.15,
    tmin=None,
    tmax=None,
    floorplan_path=None,
    layer_idx=4,
    # Matplotlib plotting parameters.
    figure_size=(5, 4),
    heatmap_axes_rect=(0.01, 0.01, 0.78, 0.98),
    colorbar_x=0.82,
    colorbar_width=0.04,
    colorbar_bottom_margin=0.02,
    colorbar_top_margin=0.02,
    colorbar_ticks=(45, 55, 65, 75, 85),
    colorbar_tick_pad=2,
    colorbar_tick_font_size=18,
    colorbar_label_font_size=18,
    max_temp_annotation_font_size=18,
    annotation_axes_margin=0.05,
):
    """
        A wrapper function to plot the thermal map produced by our HotspotManager
    """
    temperature_file = None
    floorplan_file = None

    for file in os.listdir(output_dir):
        if file.endswith('.grid.steady'):
            temperature_file = os.path.join(output_dir, file)
        if floorplan_path is None and file.endswith('L4_Chiplet.flp'):
            floorplan_file = os.path.join(output_dir, file)

    # Override floorplan file if an explicit path is provided
    if floorplan_path is not None:
        floorplan_file = floorplan_path

    if temperature_file is None:
        raise FileNotFoundError("No file ending with '.grid.steady' found in the output directory.")
    if floorplan_file is None:
        # Try to fall back to any .flp file if exactly one exists
        flp_candidates = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.endswith('.flp')
        ]
        if len(flp_candidates) == 1:
            floorplan_file = flp_candidates[0]
        elif not flp_candidates:
            raise FileNotFoundError(
                "No floorplan .flp file found in the output directory. "
                "Please provide one via --floorplan."
            )
        else:
            raise FileNotFoundError(
                "Multiple .flp files found in the output directory but none named 'L4_Chiplet.flp'. "
                "Please specify which one to use via --floorplan."
            )

    heatmap = read_heatmap(temperature_file, n_row, n_col, layer_idx=layer_idx, t_delta=t_delta)
    floorplan = read_floorplan(floorplan_file, n_row, n_col)

    plot_multi_die_heatmap(
        heatmap,
        floorplan,
        tmin=tmin,
        tmax=tmax,
        figure_size=figure_size,
        heatmap_axes_rect=heatmap_axes_rect,
        colorbar_x=colorbar_x,
        colorbar_width=colorbar_width,
        colorbar_bottom_margin=colorbar_bottom_margin,
        colorbar_top_margin=colorbar_top_margin,
        colorbar_ticks=colorbar_ticks,
        colorbar_tick_pad=colorbar_tick_pad,
        colorbar_tick_font_size=colorbar_tick_font_size,
        colorbar_label_font_size=colorbar_label_font_size,
        max_temp_annotation_font_size=max_temp_annotation_font_size,
        annotation_axes_margin=annotation_axes_margin,
    )

    plt.savefig(os.path.join(output_dir, 'thermal_map.png'))
    plt.savefig(os.path.join(output_dir, 'thermal_map.pdf'))
    plt.clf()
    plt.close()


def parse_args():
    parser = ArgumentParser(description='Plot the thermal map with chiplet outlines')
    parser.add_argument('-d', type=str, help='The output directory of the simulation')
    parser.add_argument('--n_row', type=int, default=128, help='Number of rows in the temperature grid')
    parser.add_argument('--n_col', type=int, default=128, help='Number of columns in the temperature grid')
    parser.add_argument('--t_delta', type=float, default=273.15, help='Temperature delta')
    parser.add_argument('--tmin', type=float, default=45, help='Minimum temperature')
    parser.add_argument('--tmax', type=float, default=85, help='Maximum temperature')
    parser.add_argument('--floorplan', type=str, default=None, help='Path to a specific floorplan .flp file')
    parser.add_argument('--layer_idx', type=int, default=4, help='Layer index in the .grid.steady file to visualize')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_hotspot_thermal_map(
        args.d,
        args.n_row,
        args.n_col,
        args.t_delta,
        args.tmin,
        args.tmax,
        floorplan_path=args.floorplan,
        layer_idx=args.layer_idx,
    )
