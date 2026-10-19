from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .block import Block


def visualize_floorplan(
    block_list: List[Block],
    save_path: Optional[str] = None,
    title: str = "Floorplan Visualization",
) -> None:
    """
    Visualize the block list with matplotlib.

    Args:
        block_list: List of blocks to visualize.
        save_path: Path to save the figure (optional).
        title: Title for the plot.
    """

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Define colors for different materials
    material_colors = {
        "silicon": "lightblue",
        "copper": "orange",
        "underfill": "lightgreen",
        "ubump_t16um_d25um_p50um": "lightcoral",
        "hb_t1300nm_p1750nm": "lightyellow",
        "copper_tsv": "goldenrod",
        "beol_cu_p28nm": "wheat",
    }

    # Collect materials that appear in the figure for legend
    materials_in_figure: dict[str, str] = {}

    for block in block_list:
        # Get material color
        material_name = block.material.name
        color = material_colors.get(material_name, "lightgray")
        materials_in_figure[material_name] = color

        # Create rectangle
        rect = patches.Rectangle(
            block.offset,
            block.shape[0],
            block.shape[1],
            linewidth=2,
            edgecolor="black",
            facecolor=color,
            alpha=0.7,
        )
        ax.add_patch(rect)

        # Add label
        center_x = block.offset[0] + block.shape[0] / 2
        center_y = block.offset[1] + block.shape[1] / 2
        ax.text(
            center_x,
            center_y,
            block.get_full_name(),
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=8,
        )

    # Set plot properties
    if block_list:
        max_x = max(block.offset[0] + block.shape[0] for block in block_list)
        max_y = max(block.offset[1] + block.shape[1] for block in block_list)
        ax.set_xlim(0, max_x)
        ax.set_ylim(0, max_y)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Add legend for materials that appear in the figure
    legend_patches = [
        patches.Patch(facecolor=color, edgecolor="black", alpha=0.7, label=name)
        for name, color in materials_in_figure.items()
    ]
    ax.legend(handles=legend_patches, loc="upper left", bbox_to_anchor=(1.02, 1))

    # Save figure if path is provided
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()
