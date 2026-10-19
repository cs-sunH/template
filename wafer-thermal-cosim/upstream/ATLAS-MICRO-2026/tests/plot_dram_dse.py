import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

# ── Font setup (Calibri → Carlito fallback) ────────────────────────────────
matplotlib.rcParams['font.family'] = 'sans-serif'
_avail = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
matplotlib.rcParams['font.sans-serif'] = (
    [f for f in ['Calibri', 'Carlito'] if f in _avail] + ['DejaVu Sans']
)
for _fp in sorted(Path.home().joinpath('.local', 'share', 'fonts').glob('Carlito*.ttf')):
    matplotlib.font_manager.fontManager.addfont(str(_fp))

# ── Global display settings ────────────────────────────────────────────────

# --- 2×2 small-plot fonts ---
FONT_SIZE = 22                  # tick labels, axis labels, colorbar
CELL_FONT_SIZE = 20             # cell annotations
CELL_DECIMALS = 0               # 0→"5"  1→"4.5"  2→"4.54"
CELL_SHOW_PERCENT = True        # append "%"
CELL_BOLD = True                # bold cell text

# --- Average-plot fonts (independent from 2×2) ---
AVG_FONT_SIZE = 22              # tick labels, axis labels
AVG_CELL_FONT_SIZE = 24         # cell annotations
AVG_CELL_BOLD = True           # bold cell text

# --- Panel titles: (a) OPT, (b) LLaMA, … ---
PANEL_LABEL_SIZE = 22
PANEL_LABEL_BOLD = True
PANEL_LABEL_FONT = None         # None → global font; or e.g. 'Arial'

# --- Near-max region highlighting ---
NEAR_MAX_THRESHOLD = 1        # pp gap for 6×6 plots (Layout A/B)
ROW_NEAR_MAX_THRESHOLD = 0     # pp gap for 4×1 row plot
NEAR_MAX_COLOR = '#C00000'      # border colour
NEAR_MAX_LINEWIDTH = 3        # border line width

# --- Row (4×1) plot color range ---
ROW_CMAP_RANGE = (0, 1)   # slice of Blues colormap to use (more contrast)

# --- Layout A sizing & spacing ---
LAYOUT_A_MODEL_WIDTH = 1       # width ratio for each model column
LAYOUT_A_AVG_WIDTH = 2.125         # width ratio for the average column
LAYOUT_A_ROW_GAP = 0.10        # hspace between the two model rows
LAYOUT_A_MODEL_AVG_GAP = 0.05  # gap column width ratio between models and avg

# ────────────────────────────────────────────────────────────────────────────


def _fmt_cell(val):
    if CELL_SHOW_PERCENT:
        return f'{val:.{CELL_DECIMALS}%}'
    return f'{val * 100:.{CELL_DECIMALS}f}'


def _title_kwargs():
    kw = dict(fontsize=PANEL_LABEL_SIZE,
              fontweight='bold' if PANEL_LABEL_BOLD else 'normal')
    if PANEL_LABEL_FONT:
        kw['fontfamily'] = PANEL_LABEL_FONT
    return kw


def _highlight_near_max(ax, z, threshold=None):
    """Draw outer border around cells within *threshold* pp of the max.

    Uses the *displayed* values (rounded to CELL_DECIMALS) so the visual
    region matches the numbers the reader sees in the figure.
    """
    thr = threshold if threshold is not None else NEAR_MAX_THRESHOLD
    ny, nx = z.shape
    displayed = np.round(z * 100, CELL_DECIMALS)
    max_val = displayed.max()
    mask = (max_val - displayed) <= thr

    kw = dict(color=NEAR_MAX_COLOR, lw=NEAR_MAX_LINEWIDTH,
              solid_capstyle='projecting', clip_on=False)
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


def _make_cmap(base='Blues', lo=0.0, hi=1.0):
    """Return a colormap that uses the *lo*–*hi* slice of *base*.

    lo/hi ∈ [0, 1].  Increasing *lo* removes the lightest tones;
    decreasing *hi* removes the darkest tones.  Both increase contrast.
    """
    from matplotlib.colors import LinearSegmentedColormap
    samples = plt.cm.get_cmap(base)(np.linspace(lo, hi, 256))
    return LinearSegmentedColormap.from_list('custom', samples)


def _draw_heatmap_on_ax(ax, z, x_labels, y_labels,
                        xlabel='', ylabel='', title=None,
                        show_xticks=True, show_yticks=True,
                        font_size=None, cell_font_size=None,
                        cell_bold=None,
                        vmin=0, vmax=1,
                        near_max_threshold=None,
                        cmap_range=None):
    """Render a heatmap on *ax*.  Optional overrides for font sizes."""
    fs = font_size if font_size is not None else FONT_SIZE
    cfs = cell_font_size if cell_font_size is not None else CELL_FONT_SIZE
    cb = cell_bold if cell_bold is not None else CELL_BOLD

    cmap = _make_cmap('Blues', *cmap_range) if cmap_range else 'Blues'
    im = ax.imshow(z, vmin=vmin, vmax=vmax, cmap=cmap,
                   origin='lower', aspect='equal')
    ny, nx = z.shape
    ax.set_xticks(np.arange(nx))
    ax.set_yticks(np.arange(ny))
    ax.set_xticklabels(x_labels if show_xticks else [], fontsize=fs)
    ax.set_yticklabels(y_labels if show_yticks else [], fontsize=fs)
    if not show_xticks:
        ax.tick_params(axis='x', length=0)
    if not show_yticks:
        ax.tick_params(axis='y', length=0)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fs + 2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fs + 2)
    if title:
        ax.set_title(title, **_title_kwargs())

    cell_wt = 'bold' if cb else 'normal'
    color_mid = (vmin + vmax) / 2
    for iy in range(ny):
        for ix in range(nx):
            val = z[iy, ix]
            color = 'white' if val > color_mid else 'black'
            ax.text(ix, iy, _fmt_cell(val),
                    ha='center', va='center',
                    fontsize=cfs, fontweight=cell_wt,
                    color=color)

    _highlight_near_max(ax, z, threshold=near_max_threshold)
    return im


def _add_shared_colorbar(fig, im, cbar_ax, label, font_size=None,
                         vmin=0, vmax=1):
    fs = font_size if font_size is not None else FONT_SIZE
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(label, fontsize=fs + 2)
    step = 0.2 if (vmax - vmin) > 0.35 else 0.1
    cbar.set_ticks(np.arange(vmin, vmax + 0.001, step))
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0%}'))
    cbar.ax.tick_params(labelsize=fs)
    return cbar


# ── Single-panel plot ──────────────────────────────────────────────────────

def plot_heatmap(z_data, x_labels, y_labels, xlabel, ylabel, cbar_label,
                 save_paths, figsize=(7, 5.5)):
    if isinstance(save_paths, str):
        save_paths = [save_paths]
    z = np.asarray(z_data, dtype=float)
    ny, nx = z.shape
    assert nx == 6 and ny == 6

    fig, ax = plt.subplots(figsize=figsize)
    im = _draw_heatmap_on_ax(ax, z, x_labels, y_labels,
                              xlabel=xlabel, ylabel=ylabel)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=FONT_SIZE + 2)
    cbar.set_ticks(np.arange(0, 1.01, 0.2))
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0%}'))
    cbar.ax.tick_params(labelsize=FONT_SIZE)

    plt.tight_layout()
    for p in save_paths:
        plt.savefig(p, dpi=200, bbox_inches='tight')
        print(f"Figure saved to {p}")
    plt.show()


# ── Layout A: 2×2 models + full-height average on the right ───────────────
#
#   ┌────────────────┐          ┌────────────────┐  ▌
#   │ (a)   │  (b)   │          │                │  ▌cbar
#   │───────│────────│          │  (e) Average   │  ▌
#   │ (c)   │  (d)   │          │                │  ▌
#   └────────────────┘          └────────────────┘  ▌
#
#   Average data area is post-aligned with the 2×2 block so that
#   the top/bottom heatmap edges match exactly.

def plot_combined_layout_A(panels, x_labels, y_labels, xlabel, ylabel,
                           cbar_label, save_paths, figsize=(20, 9)):
    if isinstance(save_paths, str):
        save_paths = [save_paths]
    assert len(panels) == 5

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 5,
        width_ratios=[LAYOUT_A_MODEL_WIDTH, LAYOUT_A_MODEL_WIDTH,
                      LAYOUT_A_MODEL_AVG_GAP, LAYOUT_A_AVG_WIDTH, 0.08],
        hspace=LAYOUT_A_ROW_GAP, wspace=0.12)

    model_axes, im = [], None
    cfg = [
        (0, 0, False, True,  False, True),   # (a) top-L
        (0, 1, False, False, False, False),  # (b) top-R
        (1, 0, True,  True,  True,  True),   # (c) bot-L
        (1, 1, True,  False, True,  False),  # (d) bot-R
    ]
    for idx, (r, c, sxt, syt, sxl, syl) in enumerate(cfg):
        ax = fig.add_subplot(gs[r, c])
        title, z = panels[idx]
        im = _draw_heatmap_on_ax(
            ax, np.asarray(z, dtype=float), x_labels, y_labels,
            xlabel=xlabel if sxl else '', ylabel=ylabel if syl else '',
            # title=f'({chr(97 + idx)}) {title}',
            title=f'{title}',
            show_xticks=sxt, show_yticks=syt)
        model_axes.append(ax)

    # average (initial placement via gridspec, aligned later)
    ax_avg = fig.add_subplot(gs[:, 3])
    t, z_avg = panels[4]
    im = _draw_heatmap_on_ax(
        ax_avg, np.asarray(z_avg, dtype=float), x_labels, y_labels,
        xlabel=xlabel, ylabel=ylabel,
        title=f'(e) {t}',
        show_xticks=True, show_yticks=True,
        font_size=AVG_FONT_SIZE,
        cell_font_size=AVG_CELL_FONT_SIZE,
        cell_bold=AVG_CELL_BOLD)

    # colorbar in dedicated column (no space-stealing)
    cbar_ax = fig.add_subplot(gs[:, 4])
    _add_shared_colorbar(fig, im, cbar_ax, cbar_label)

    # ── post-draw alignment: match avg data area to 2×2 block ──
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    ext_a = model_axes[0].get_window_extent(renderer).transformed(inv)
    ext_c = model_axes[2].get_window_extent(renderer).transformed(inv)
    ext_avg = ax_avg.get_window_extent(renderer).transformed(inv)
    pos_avg = ax_avg.get_position()

    pad_top = pos_avg.y1 - ext_avg.y1
    pad_bot = ext_avg.y0 - pos_avg.y0
    new_y1 = ext_a.y1 + pad_top
    new_y0 = ext_c.y0 - pad_bot
    ax_avg.set_position([pos_avg.x0, new_y0, pos_avg.width, new_y1 - new_y0])

    for p in save_paths:
        plt.savefig(p, dpi=200, bbox_inches='tight')
        print(f"Figure saved to {p}")
    plt.show()


# ── Layout B: 2×2 models + full-width average at bottom ───────────────────

def plot_combined_layout_B(panels, x_labels, y_labels, xlabel, ylabel,
                           cbar_label, save_paths, figsize=(10, 13)):
    if isinstance(save_paths, str):
        save_paths = [save_paths]
    assert len(panels) == 5

    fig = plt.figure(figsize=figsize)
    hs = PANEL_LABEL_SIZE / 72 * 3.0 / (figsize[1] / 3)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1],
                          width_ratios=[1, 1, 0.05],
                          hspace=hs, wspace=0.15)

    all_axes, im = [], None
    cfg = [
        (0, 0, False, True,  False, True),
        (0, 1, False, False, False, False),
        (1, 0, False, True,  False, True),
        (1, 1, False, False, False, False),
    ]
    for idx, (r, c, sxt, syt, sxl, syl) in enumerate(cfg):
        ax = fig.add_subplot(gs[r, c])
        title, z = panels[idx]
        im = _draw_heatmap_on_ax(
            ax, np.asarray(z, dtype=float), x_labels, y_labels,
            xlabel=xlabel if sxl else '', ylabel=ylabel if syl else '',
            title=f'({chr(97 + idx)}) {title}',
            show_xticks=sxt, show_yticks=syt)
        all_axes.append(ax)

    ax_avg = fig.add_subplot(gs[2, 0:2])
    t, z_avg = panels[4]
    im = _draw_heatmap_on_ax(
        ax_avg, np.asarray(z_avg, dtype=float), x_labels, y_labels,
        xlabel=xlabel, ylabel=ylabel,
        title=f'(e) {t}',
        show_xticks=True, show_yticks=True,
        font_size=AVG_FONT_SIZE,
        cell_font_size=AVG_CELL_FONT_SIZE,
        cell_bold=AVG_CELL_BOLD)
    all_axes.append(ax_avg)

    cbar_ax = fig.add_subplot(gs[:, 2])
    _add_shared_colorbar(fig, im, cbar_ax, cbar_label)

    for p in save_paths:
        plt.savefig(p, dpi=200, bbox_inches='tight')
        print(f"Figure saved to {p}")
    plt.show()


# ── Row layout: N panels in a single row ──────────────────────────────────
#
#   ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ▌
#   │ (a)   │ │ (b)   │ │ (c)   │ │ (d)   │ ▌cbar
#   └───────┘ └───────┘ └───────┘ └───────┘ ▌

def plot_row_layout(panels, x_labels, y_labels, xlabel, ylabel,
                    cbar_label, save_paths, figsize=None,
                    vmin=0, vmax=1, near_max_threshold=None,
                    cmap_range=None):
    if isinstance(save_paths, str):
        save_paths = [save_paths]
    n = len(panels)
    if figsize is None:
        figsize = (5 * n + 1, 5)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1] * n + [0.05],
                          wspace=0.15)

    all_axes, im = [], None
    for idx in range(n):
        ax = fig.add_subplot(gs[0, idx])
        title, z = panels[idx]
        z = np.asarray(z, dtype=float)
        is_left = (idx == 0)
        im = _draw_heatmap_on_ax(
            ax, z, x_labels, y_labels,
            xlabel=xlabel,
            ylabel=ylabel if is_left else '',
            # title=f'({chr(97 + idx)}) {title}',
            title=f'{title}',
            show_xticks=True,
            show_yticks=is_left,
            vmin=vmin, vmax=vmax,
            near_max_threshold=near_max_threshold,
            cmap_range=cmap_range)
        all_axes.append(ax)

    cbar_ax = fig.add_subplot(gs[0, n])
    cmap = _make_cmap('Blues', *cmap_range) if cmap_range else 'Blues'
    cbar = _add_shared_colorbar(fig, im, cbar_ax, cbar_label,
                                vmin=vmin, vmax=vmax)

    for p in save_paths:
        plt.savefig(p, dpi=200, bbox_inches='tight')
        print(f"Figure saved to {p}")
    plt.show()


# ── Data ────────────────────────────────────────────────────────────────────

datasets = {
    "OPT": [
        [0.0454,   0.085772, 0.164755, 0.254743, 0.337434, 0.393613],
        [0.096156, 0.176113, 0.297238, 0.409095, 0.48998,  0.565573],
        [0.209054, 0.325011, 0.46274,  0.518381, 0.60133,  0.651082],
        [0.344992, 0.455292, 0.533943, 0.600257, 0.671354, 0.701632],
        [0.483413, 0.531434, 0.59355,  0.654334, 0.71183,  0.766365],
        [0.543446, 0.592044, 0.656241, 0.69946,  0.748085, 0.803714],
    ],
    "LLaMA": [
        [0.052446, 0.101755, 0.195397, 0.313568, 0.422698, 0.526689],
        [0.108597, 0.212029, 0.352434, 0.493887, 0.572191, 0.632778],
        [0.229727, 0.378094, 0.525518, 0.614845, 0.65676,  0.746694],
        [0.344821, 0.527811, 0.617931, 0.678509, 0.748853, 0.802069],
        [0.47231,  0.522626, 0.590897, 0.6489,   0.6766,   0.701998],
        [0.535776, 0.594562, 0.639272, 0.671682, 0.692166, 0.712395],
    ],
    "Mixtral": [
        [0.056876, 0.082624, 0.161939, 0.270117, 0.407403, 0.535825],
        [0.085684, 0.170861, 0.294307, 0.461722, 0.545646, 0.649307],
        [0.180282, 0.310668, 0.486132, 0.576607, 0.681603, 0.743831],
        [0.272104, 0.438763, 0.511127, 0.588764, 0.638617, 0.662598],
        [0.369227, 0.418003, 0.46847,  0.498148, 0.506045, 0.522264],
        [0.417773, 0.467368, 0.496714, 0.513815, 0.516469, 0.528295],
    ],
    "DeepSeek": [
        [0.056876, 0.082624, 0.161939, 0.270117, 0.407403, 0.535825],
        [0.085684, 0.170861, 0.294307, 0.461722, 0.545646, 0.649307],
        [0.180282, 0.310668, 0.486132, 0.576607, 0.681603, 0.743831],
        [0.272104, 0.438763, 0.511127, 0.588764, 0.638617, 0.662598],
        [0.369227, 0.418003, 0.46847,  0.498148, 0.506045, 0.522264],
        [0.417773, 0.467368, 0.496714, 0.513815, 0.516469, 0.528295],
    ],
    "avg_consider_latency": [
        [0.05736915,  0.101391062, 0.198014668, 0.312339,    0.43378995,  0.53354884],
        [0.106707913, 0.208697532, 0.341612086, 0.488538872, 0.57137211,  0.65130284],
        [0.225723294, 0.363237827, 0.519019272, 0.600823854, 0.677063765, 0.745924977],
        [0.342817172, 0.499582216, 0.579838386, 0.649692161, 0.71186,     0.750732996],
        [0.463821692, 0.520684622, 0.581073154, 0.62897941,  0.657105035, 0.680189645],
        [0.522960794, 0.581361077, 0.626539058, 0.657380297, 0.672570398, 0.690496209],
    ],
    "geomean": [
        [0.055836816, 0.100576368, 0.196229868, 0.307082105, 0.422506692, 0.515892195],
        [0.10650315,  0.206813602, 0.337751435, 0.478517221, 0.560641782, 0.638810484],
        [0.226088616, 0.360284522, 0.512019838, 0.589927492, 0.665122202, 0.731917621],
        [0.346331387, 0.495683309, 0.576146441, 0.644600655, 0.708401509, 0.746894313],
        [0.470406482, 0.526304879, 0.587516586, 0.63914285,  0.67225303,  0.699625686],
        [0.530390231, 0.587309642, 0.636750379, 0.670766571, 0.690807983, 0.714021736],
    ],
    "avg_no_latency": [
        [0.05662475,  0.10301225, 0.2016265,  0.312639,  0.42898175, 0.52344675],
        [0.108559,    0.2114355,  0.3415175,  0.481683,  0.56340925, 0.64107325],
        [0.230211,    0.363782,   0.51394525, 0.59221375, 0.66668175, 0.73379925],
        [0.3515935,   0.49860575, 0.579096,   0.646881,  0.71080225, 0.7502185],
        [0.47644675,  0.5332385,  0.5945175,  0.647586,  0.68311375, 0.71083425],
        [0.53689425,  0.5942935,  0.6452815,  0.680888,  0.70207425, 0.725927],
    ],
    "Qwen": [
        [0.07254846, 0.14117042, 0.27448379, 0.40533936, 0.54669435, 0.62781847],
        [0.14364283, 0.27566469, 0.40553409, 0.54435865, 0.6232125,  0.68337617],
        [0.31218319, 0.44611565, 0.57479746, 0.65463523, 0.71216295, 0.77942535],
        [0.44649726, 0.5462854,  0.64648626, 0.69323369, 0.74983815, 0.81015179],
        [0.54185527, 0.6041426,  0.69025609, 0.74671266, 0.80608824, 0.8151069],
        [0.59677281, 0.64641012, 0.74671266, 0.80608824, 0.8151069,  0.82246839],
    ],
    "avg_consider_latency_Qwen": [
        [0.054549337, 0.094200982, 0.182677305, 0.293073662, 0.410871773, 0.51049763],
        [0.099875668, 0.193958822, 0.32387536,  0.470306198, 0.552071125, 0.63194676],
        [0.21231799,  0.347405537, 0.504022646, 0.586441791, 0.662830858, 0.731899653],
        [0.323202709, 0.48028787,  0.562201431, 0.629885414, 0.689964777, 0.727475997],
        [0.436761789, 0.487281027, 0.54843383,  0.592033178, 0.617175847, 0.640347984],
        [0.492135834, 0.544942078, 0.589559444, 0.618294631, 0.632916377, 0.651844574],
    ],
    "geomean_Qwen": [
        [0.055986248, 0.100447193, 0.19449398,  0.305809641, 0.422179324, 0.513890022],
        [0.106474222, 0.204787145, 0.334389416, 0.474711077, 0.555669586, 0.631266131],
        [0.228012208, 0.361252163, 0.510561922, 0.588943321, 0.661694628, 0.728629563],
        [0.346728159, 0.489896698, 0.574620017, 0.638525783, 0.700469195, 0.741368977],
        [0.462307091, 0.514623722, 0.580316984, 0.630408836, 0.66576353,  0.691781668],
        [0.519065617, 0.571057137, 0.628061598, 0.664169594, 0.683289913, 0.706243605],
    ],
    "avg_Qwen": [
        [0.056817615, 0.102830355, 0.199143698, 0.31094184,  0.428557338, 0.520986368],
        [0.108519958, 0.208666923, 0.337378273, 0.477265663, 0.557757375, 0.632758543],
        [0.232811548, 0.364972163, 0.512296865, 0.591117058, 0.662963988, 0.730258088],
        [0.352103565, 0.49203785,  0.577371815, 0.640190923, 0.702165538, 0.744112698],
        [0.466701318, 0.5190514,   0.585793273, 0.637023665, 0.67514081,  0.701433475],
        [0.523441953, 0.57509603,  0.634734915, 0.67276131,  0.692956725, 0.716718098],
    ],
}

if __name__ == "__main__":
    import os

    output_dir = os.path.join(os.path.dirname(__file__),
                              '..', 'results', 'dram_dse')
    os.makedirs(output_dir, exist_ok=True)

    x_labels = ['2', '4', '8', '16', '32', '64']
    y_labels = ['2', '4', '8', '16', '32', '64']
    common_kw = dict(x_labels=x_labels, y_labels=y_labels,
                     xlabel='Row Size (KB)', ylabel='Channel Number',
                     cbar_label='Bandwidth Utilization')

    combined_panels = [
        ("OPT",     datasets["OPT"]),
        ("LLaMA",   datasets["LLaMA"]),
        ("Mixtral",  datasets["Mixtral"]),
        ("Qwen",    datasets["Qwen"]),
        ("Average", datasets["avg_consider_latency_Qwen"]),
    ]

    plot_combined_layout_A(
        combined_panels,
        save_paths=[os.path.join(output_dir, 'combined_A.pdf'),
                    os.path.join(output_dir, 'combined_A.png')],
        **common_kw)

    plot_combined_layout_B(
        combined_panels,
        save_paths=[os.path.join(output_dir, 'combined_B.pdf'),
                    os.path.join(output_dir, 'combined_B.png')],
        **common_kw)

    # ── 4×4 row layout ──
    row_panels = [
        ("OPT", [
            [0.488381598, 0.567248537, 0.744564056, 0.777361069],
            [0.552034736, 0.676647144, 0.777246354, 0.77731888],
            [0.632898326, 0.775411733, 0.780009703, 0.792304891],
            [0.72491069,  0.779100802, 0.785572626, 0.793084179],
        ]),
        ("LLaMA", [
            [0.538186669, 0.609192168, 0.693238727, 0.754647498],
            [0.587985717, 0.69898444,  0.755572761, 0.801894615],
            [0.67870541,  0.760809514, 0.804276356, 0.808350252],
            [0.756569707, 0.802830019, 0.814744011, 0.823643457],
        ]),
        ("PaLM", [
            [0.483366042, 0.559995789, 0.735938479, 0.770325392],
            [0.541758896, 0.672100109, 0.771096075, 0.766375913],
            [0.617805525, 0.773717757, 0.768996052, 0.783503424],
            [0.71061188,  0.767858269, 0.773444269, 0.785003728],
        ]),
        ("Average", [
            [0.50405374,  0.579869347, 0.723674859, 0.767032053],
            [0.561700453, 0.683346151, 0.767546693, 0.783193887],
            [0.644546034, 0.769564526, 0.785739734, 0.795637756],
            [0.7321901,   0.784577861, 0.792643128, 0.801926396],
        ]),
    ]
    row_x = ['8', '16', '32', '64']
    row_y = ['8', '16', '32', '64']
    plot_row_layout(
        row_panels, row_x, row_y,
        xlabel='Row Size (KB)', ylabel='Channel Number',
        cbar_label='Bandwidth Utilization',
        save_paths=[os.path.join(output_dir, 'row_4x4.pdf'),
                    os.path.join(output_dir, 'row_4x4.png')],
        vmin=0.4, vmax=1,
        near_max_threshold=ROW_NEAR_MAX_THRESHOLD,
        cmap_range=ROW_CMAP_RANGE)
